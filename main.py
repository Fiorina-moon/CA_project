"""
CA Project - 骨骼动画系统主入口

用法: python main.py <命令> [参数]

命令:
  export <动画名>    导出动画视频
  compute           重新计算蒙皮权重
  list              列出所有可用动画
  help              显示帮助信息

示例:
  python main.py export  walk_circle --angle 180 --fps 60
  python main.py compute
  python main.py compute --max-influences 6
  python main.py list
"""
import sys
from pathlib import Path

# 添加 src 到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.config import (
    ELK_OBJ_PATH, SKELETON_JSON_PATH, WEIGHTS_DIR, 
    ANIMATIONS_DIR,
)
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.skinning.weight_calculator import WeightCalculator
from src.utils.file_io import save_weights_npz
from src.rendering.video_export import VideoExporter


def show_help():
    """显示帮助信息"""
    print(__doc__)


def list_animations():
    """列出所有可用动画"""
    print(f"\n动画文件夹: {ANIMATIONS_DIR}")
    print("=" * 60)
    
    animations = list(ANIMATIONS_DIR.glob("*.json"))
    if not animations:
        print("未找到动画文件")
        print(f"请将动画文件放到: {ANIMATIONS_DIR}")
        return
    
    print(f"找到 {len(animations)} 个动画:\n")
    for i, anim_file in enumerate(sorted(animations), 1):
        print(f"  {i}. {anim_file.stem}")
    
    print("\n" + "=" * 60)
    print("提示: 使用 'python main.py export <动画名>' 导出视频")
    print("=" * 60)


def compute_weights(args):
    """
    重新计算蒙皮权重
    
    参数格式:
      [--max-influences 数量]
    """
    # 解析参数
    max_influences = 4  # 默认值
    
    try:
        if '--max-influences' in args:
            idx = args.index('--max-influences')
            if idx + 1 < len(args):
                max_influences = int(args[idx + 1])
                if max_influences < 1 or max_influences > 8:
                    print("警告: max_influences 应在 1-8 之间，使用默认值 4")
                    max_influences = 4
    except Exception as e:
        print(f"警告: 参数解析错误 ({e})，使用默认值")
    
    # 执行权重计算
    recompute_weights(max_influences)


def recompute_weights(max_influences=4):
    """
    使用区域分割算法重新计算权重
    
    Args:
        max_influences: 每个顶点的最大影响骨骼数量
    """
    print("=" * 60)
    print("重新计算蒙皮权重 (区域分割版)")
    print("=" * 60)
    
    try:
        # 加载数据
        print("\n[1/4] 加载模型和骨架...")
        mesh = OBJLoader.load(ELK_OBJ_PATH)
        skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
        
        print(f"  模型: {mesh.get_vertex_count()} 顶点")
        print(f"  骨架: {skeleton.get_bone_count()} 骨骼")
        
        # 打印骨骼名称，帮助调试区域分类
        print(f"\n  骨骼列表:")
        for i, bone in enumerate(skeleton.bones):
            print(f"    [{i:2d}] {bone.name}")
        
        # 计算权重
        print(f"\n[2/4] 计算权重 (最大影响数: {max_influences})...")
        calculator = WeightCalculator(max_influences=max_influences)
        weights = calculator.compute_weights(mesh, skeleton)
        
        # 备份旧文件
        print("\n[3/4] 备份旧权重文件...")
        old_weights_path = WEIGHTS_DIR / "elk_weights.npz"
        if old_weights_path.exists():
            backup_path = WEIGHTS_DIR / "elk_weights_backup.npz"
            import shutil
            shutil.copy(old_weights_path, backup_path)
            print(f"  已备份到: {backup_path}")
        else:
            print("  未找到旧权重文件，跳过备份")
        
        # 保存新权重
        print("\n[4/4] 保存新权重...")
        save_weights_npz(weights, old_weights_path)
        print(f"  已保存到: {old_weights_path}")
        
        print("\n" + "=" * 60)
        print("权重重新计算完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n错误: 权重计算失败 - {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_export_args(args):
    """
    解析导出命令的参数
    
    Args:
        args: 参数列表
    
    Returns:
        tuple: (animation_name, view_angle, render_mode, fps, duration)
    """
    if len(args) < 1:
        print("错误: 请指定动画名称")
        print("\n用法: python main.py export <动画名> [选项]")
        print("\n可选参数:")
        print("  --angle <度数>    相机方位角 (默认: 90)")
        print("                    0=后面, 90=右侧, 180=前面, 270=左侧")
        print("  --mode <模式>     渲染模式 (默认: transparent)")
        print("                    solid, wireframe, transparent, wireframe_transparent")
        print("  --fps <帧率>      视频帧率 (默认: 30)")
        print("  --duration <秒>   导出时长 (默认: 0=完整动画)")
        return None
    
    # 提取动画名称
    anim_name = args[0]
    
    # 解析可选参数
    view_angle = 90
    render_mode = 'transparent_with_wireframe'
    fps = 30
    duration = 0
    
    try:
        if '--angle' in args:
            idx = args.index('--angle')
            if idx + 1 < len(args):
                view_angle = int(args[idx + 1])
        
        if '--mode' in args:
            idx = args.index('--mode')
            if idx + 1 < len(args):
                mode = args[idx + 1]
                if mode in ['solid', 'wireframe', 'transparent', 'wireframe_transparent']:
                    render_mode = mode
                else:
                    print(f"警告: 未知渲染模式 '{mode}'，使用默认值 'transparent'")
        
        if '--fps' in args:
            idx = args.index('--fps')
            if idx + 1 < len(args):
                fps = int(args[idx + 1])
        
        if '--duration' in args:
            idx = args.index('--duration')
            if idx + 1 < len(args):
                duration = float(args[idx + 1])
    except Exception as e:
        print(f"警告: 参数解析错误 ({e})，使用默认值")
    
    return anim_name, view_angle, render_mode, fps, duration


def export_video_command(args):
    """
    处理 export 命令
    
    Args:
        args: 命令行参数列表
    """
    parsed = parse_export_args(args)
    if parsed is None:
        return
    
    anim_name, view_angle, render_mode, fps, duration = parsed
    
    # 调用导出函数
    export_video(anim_name, view_angle, render_mode, fps, duration)


def export_video(animation_name, view_angle=90, render_mode='transparent', fps=30, duration=0):
    """
    导出动画为视频文件
    
    Args:
        animation_name: 动画名称（不含.json）
        view_angle: 相机方位角（0=后, 90=右, 180=前, 270=左）
        render_mode: 渲染模式（solid, wireframe, transparent, wireframe_transparent）
        fps: 视频帧率
        duration: 导出时长（0=完整动画）
    
    Returns:
        bool: 是否成功
    """
    exporter = VideoExporter(
        mesh_path=ELK_OBJ_PATH,
        skeleton_path=SKELETON_JSON_PATH,
        weights_path=WEIGHTS_DIR / "elk_weights.npz",
        animations_dir=ANIMATIONS_DIR
    )
    
    return exporter.export(
        animation_name=animation_name,
        view_angle=view_angle,
        render_mode=render_mode,
        fps=fps,
        duration=duration
    )


def main():
    """主函数 - 解析命令行参数并执行相应操作"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command in ['help', '-h', '--help']:
        show_help()
    
    elif command == 'export':
        export_video_command(sys.argv[2:])
    
    elif command == 'compute':
        compute_weights(sys.argv[2:])
    
    elif command == 'list':
        list_animations()
    
    else:
        print(f"错误: 未知命令 '{command}'")
        print("使用 'python main.py help' 查看帮助")


if __name__ == "__main__":
    main()
