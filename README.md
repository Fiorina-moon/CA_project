
## 项目结构

```
CA_project/
├── src/
│   ├── core/              # 核心数据结构
│   │   ├── mesh.py           # 网格模型
│   │   ├── mesh_loader.py    # OBJ文件加载
│   │   ├── skeleton.py       # 骨架系统
│   │   └── skeleton_loader.py # 骨架加载
│   │
│   ├── animation/         # 动画系统
│   │   ├── keyframe.py       # 关键帧定义
│   │   ├── interpolation.py  # 插值算法
│   │   └── animator.py       # 动画播放控制
│   │
│   ├── skinning/          # 蒙皮系统
│   │   ├── weight_calculator.py # 权重计算
│   │   └── deformer.py       # 网格变形（LBS）
│   │
│   ├── rendering/         # 渲染模块
│   │   ├── renderer.py       # OpenGL渲染器
│   │   └── video_export.py   # 视频导出
│   │
│   ├── utils/             # 工具模块
│   │   ├── math_utils.py     # 数学工具
│   │   ├── geometry.py       # 几何计算
│   │   └── file_io.py        # 文件读写
│   │
│   ├── config.py          # 配置文件
│   └── main.py            # 主程序入口
│
└── data/                  # 数据目录
    ├── models/            # 3D模型（.obj）
    ├── skeletons/         # 骨架文件（.json）
    ├── animations/        # 动画文件（.json）
    └── weights/           # 权重文件（.npz）
```

---
## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例


#### **列出所有可用动画**
```bash
python main.py list
```

---

####  **计算蒙皮权重**

**基础用法（使用默认参数）**：
```bash
python main.py compute
```

**指定最大影响骨骼数**：
```bash
# 每个顶点最多受4个骨骼影响（默认）
python main.py compute --max-influences 4
```

---

####  **导出动画视频**

**基础用法（使用默认参数）**：
```bash
python main.py export walk_circle
```
- 默认视角：90°（右侧）
- 默认模式：半透明+线框
- 默认帧率：30 FPS
- 默认时长：完整动画
---

**完整命令示例**：
```bash

python main.py export run --angle 90 --mode wireframe_transparent --fps 30 --duration 5
```
