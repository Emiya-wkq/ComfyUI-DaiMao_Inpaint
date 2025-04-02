from .inpaint import DemoInpaint
from .mask_overlay import MaskOverlay

NODE_CLASS_MAPPINGS = {
    "呆毛Demo_Inpainting": DemoInpaint,
    "遮罩叠加处理": MaskOverlay
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "呆毛Demo_Inpainting": "呆毛Demo Inpainting",
    "遮罩叠加处理": "呆毛Demo-遮罩叠加处理"
} 