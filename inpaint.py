import torch
import comfy.utils
import comfy.sample
from nodes import KSampler, SetLatentNoiseMask, KSamplerAdvanced

class DemoInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "positive_cond": ("CONDITIONING",),
                "negative_cond": ("CONDITIONING",),
                "cfg": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 30.0,
                    "step": 0.5,
                    "display_name": "CFG Scale"
                }),
                # 修改参数名称和默认值
                "non_redraw_strength": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 0.5, 
                    "step": 0.01,
                    "display_name": "非重绘区域强度"
                }),
                "redraw_strength": ("FLOAT", {
                    "default": 0.9, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display_name": "重绘区域强度"
                }),
                # 新增步数参数
                "steps": ("INT", {
                    "default": 30, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1,
                    "display_name": "总步数"
                }),
                # 修改画质修复参数为布尔值
                "quality_fix": ("BOOLEAN", {
                    "default": True,
                    "display_name": "画质修复"
                }),
                # 新增用户选择参数
                "use_flux_config": (["enable", "disable"], {
                    "default": "disable",
                    "display_name": "使用Flux配置"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "custom_nodes"

    def execute(self, model, vae, image, mask, positive_cond, negative_cond,
                non_redraw_strength, redraw_strength, steps, quality_fix, seed, cfg, use_flux_config):
        
        # 固定参数设置
        sampler_name = 0  # euler
        scheduler = 3     # 调度器类型
        
        # 计算每个阶段的步数（向上取整）
        import math
        phase1_steps = math.ceil(steps / 2)
        phase2_steps = steps - phase1_steps
        
        # 获取设备
        device = model.model.device if hasattr(model, 'model') else model.device
        
        # 动态参数设置
        model_name = getattr(model, 'name', '').lower()
        # 动态参数设置（根据用户选择）
        is_flux_model = (use_flux_config == "enable")

        if is_flux_model:
            sampler_name = "euler"
            # scheduler = 3  # 保持原有调度器设置
            scheduler = 'sgm_uniform'
        else:
            sampler_name = "uni_pc"
            scheduler = 'karras'  # 使用karras调度器

        
        # 编码图像到潜在空间
        from nodes import VAEEncode
        vae_encoder = VAEEncode()
        latent_dict = vae_encoder.encode(vae, image)[0]
        input_latent = latent_dict["samples"].to(device)
        
        # 处理mask
        mask = mask.float().to(device)
        
        # 将mask调整到潜在空间尺寸
        mask_resized = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
            size=(input_latent.shape[2], input_latent.shape[3]), 
            mode='bilinear'
        )
        
        # 计算mask强度梯度（核心修改部分）
        # mask值为0时应用非重绘区域强度，mask值为1时应用重绘区域强度
        mask_strength = mask_resized * (redraw_strength - non_redraw_strength) + non_redraw_strength
        
        # 应用噪声遮罩
        noise_mask = SetLatentNoiseMask()
        latent_with_mask = noise_mask.set_mask({"samples": input_latent}, mask_strength)[0]
        
        # 三阶段采样流程
        advanced_sampler = KSamplerAdvanced()
        
        # === 第一阶段采样 (使用 phase1_steps) ===
        result = advanced_sampler.sample(
            model=model,
            add_noise=0.00,
            noise_seed=seed,
            steps=steps,  # 总步数
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_cond,
            negative=negative_cond,
            latent_image=latent_with_mask,
            start_at_step=0,
            end_at_step=phase1_steps,  # 使用计算出的第一阶段步数
            return_with_leftover_noise=False
        )[0]
        samples = result["samples"].to(device)
        
        
        # 生成二值化遮罩（0或1）
        binary_mask = (mask_resized >= 0.5).float()
        
        # 构建新遮罩：非重绘区域保持第一阶段非重绘强度
        phase2_mask = binary_mask * 1.0 + (1 - binary_mask) * non_redraw_strength
        
        # 应用新遮罩到第一阶段输出
        latent_phase2 = noise_mask.set_mask(
            {"samples": samples},
            phase2_mask
        )[0]
     
        # === 第二阶段采样 (使用 phase2_steps) ===
        result = advanced_sampler.sample(
            model=model,
            add_noise=0.00,
            noise_seed=seed + 1,
            steps=steps,  # 总步数
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive_cond,
            negative=negative_cond,
             # [!] 使用新处理的潜在变量 ------------------
            latent_image=latent_phase2,
            start_at_step=phase1_steps,  # 从第一阶段结束的步数开始
            end_at_step=steps,  # 到总步数结束
            return_with_leftover_noise=False
        )[0]
        samples = result["samples"].to(device)
        
        # === 最终微调采样 ===
        # 根据画质修复参数决定是否执行
        if quality_fix:  # 直接使用布尔值判断
            sampler = KSampler()
            result = sampler.sample(
                model,
                seed + 2,
                5,
                cfg,
                sampler_name,
                scheduler,
                positive_cond,
                negative_cond,
                {"samples": samples},
                0.1
            )[0]
            samples = result["samples"].to(device)
        
        # 解码潜在空间到图像
        from nodes import VAEDecode
        vae_decoder = VAEDecode()
        final_latent_dict = {"samples": samples}
        images = vae_decoder.decode(vae, final_latent_dict)[0]
        
        return (images,)