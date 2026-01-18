"""
全局配置文件
"""
import os
from pathlib import Path

# ============ 路径配置 ============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 数据路径
MODELS_DIR = DATA_DIR / "models"
SKELETON_DIR = DATA_DIR / "skeleton"
ANIMATIONS_DIR = DATA_DIR / "animations"
WEIGHTS_DIR = DATA_DIR / "weights"

# 输出路径
FRAMES_DIR = OUTPUT_DIR / "frames"
VIDEOS_DIR = OUTPUT_DIR / "videos"
DEBUG_DIR = OUTPUT_DIR / "debug"

# 模型文件
ELK_OBJ_PATH = MODELS_DIR / "elk.obj"
SKELETON_JSON_PATH = SKELETON_DIR / "skeleton.json"

# 确保目录存在
for directory in [ANIMATIONS_DIR, WEIGHTS_DIR, FRAMES_DIR, VIDEOS_DIR, DEBUG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============ 渲染配置 ============
RENDER_CONFIG = {
    "width": 800,
    "height": 600,
    "fps": 30,
    "background_color": (0.2, 0.2, 0.2, 1.0)
}

# ============ 调试配置 ============
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
