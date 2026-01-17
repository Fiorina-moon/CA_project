"""
导出动画视频
用法: python export_video.py <动画名> [--angle 90] [--mode solid] [--fps 30] [--duration 0]
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import ELK_OBJ_PATH, SKELETON_JSON_PATH, WEIGHTS_DIR, ANIMATIONS_DIR, FRAMES_DIR, VIDEOS_DIR
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.skinning.deformer import SkinDeformer
from src.animation.animator import Animator
from src.rendering.renderer import Renderer
from src.rendering.frame_exporter import FrameExporter
from src.utils.file_io import load_weights_npz, load_animation
from src.utils.math_utils import Vector3


def export_animation_video(animation_name, view_angle=90, render_mode='solid', fps=30, duration=0):
    """导出动画为视频文件"""
    print(f"\n{'='*60}")
    print(f"导出动画视频: {animation_name}")
    print(f"渲染模式: {render_mode}")
    print(f"视角: {view_angle}°")
    print(f"帧率: {fps} FPS")
    print('='*60)
    
    # 加载数据
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    weights = load_weights_npz(WEIGHTS_DIR / "elk_weights.npz")
    
    # 构建动画文件路径
    if not animation_name.endswith('.json'):
        animation_name += '.json'
    
    anim_path = ANIMATIONS_DIR / animation_name
    
    if not anim_path.exists():
        print(f"❌ 文件不存在: {anim_path}")
        return False
    
    animation = load_animation(anim_path)
    
    print(f"\n动画信息:")
    print(f"  名称: {animation.name}")
    print(f"  时长: {animation.duration}秒")
    
    # 创建动画系统
    deformer = SkinDeformer(mesh, skeleton, weights)
    animator = Animator(skeleton)
    animator.load_clip(animation)
    animator.play()
    
    # 创建渲染器
    renderer = Renderer(800, 600, f"Exporting - {animation.name}")
    if not renderer.initialize():
        return False
    
    # 设置渲染模式
    renderer.render_mode = render_mode
    
    # 相机设置
    min_pos, max_pos = mesh.get_bounding_box()
    center = Vector3(
        (min_pos.x + max_pos.x) / 2,
        (min_pos.y + max_pos.y) / 2,
        (min_pos.z + max_pos.z) / 2
    )
    size = max(max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z)
    
    adjusted_center = Vector3(
        center.x,           # X 不变
        center.y - 1.2,     # Y 向后偏移（调整前后位置）
        center.z + 1.2     # Z 向上偏移（调整上下位置）
    )
    renderer.camera.target = adjusted_center
    renderer.camera.distance = size * 2.0
    renderer.camera.azimuth = view_angle
    renderer.camera.elevation = 0
    
    print(f"\n📷 相机设置:")
    print(f"   方位角: {view_angle}°")
    print(f"   距离: {renderer.camera.distance:.2f}")
    
    # 创建帧导出器
    exporter = FrameExporter(800, 600)
    
    # 计算导出参数
    if duration <= 0:
        duration = animation.duration  # 使用动画的完整时长
    
    total_frames = int(duration * fps)
    dt = 1.0 / fps
    
    print(f"\n导出设置:")
    print(f"  总时长: {duration}秒")
    print(f"  总帧数: {total_frames}")
    print(f"  每帧间隔: {dt:.4f}秒")
    
    # 清空旧帧
    print(f"\n🗑️  清理旧帧...")
    for old_frame in FRAMES_DIR.glob("frame_*.png"):
        old_frame.unlink()
    
    print(f"\n🎬 开始渲染帧...")
    
    for frame_idx in range(total_frames):
        # 显示进度
        if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
            progress = (frame_idx + 1) / total_frames * 100
            print(f"  进度: {frame_idx + 1}/{total_frames} ({progress:.1f}%) - 动画时间: {animator.get_current_time():.2f}s")
        
        # 更新动画
        animator.update(dt)
        deformer.update()
        
        # 渲染
        renderer.render_frame(mesh, deformer, skeleton)
        
        # 捕获帧
        image = exporter.capture_frame()
        frame_path = FRAMES_DIR / f"frame_{frame_idx:04d}.png"
        exporter.save_frame(image, frame_path)
        
        renderer.poll_events()
    
    renderer.cleanup()
    
    print(f"\n✓ 帧渲染完成: {total_frames} 帧")
    
    # 合成视频
    output_name = animation.name.replace(' ', '_')
    output_video = VIDEOS_DIR / f"{output_name}.mp4"
    
    print(f"\n🎞️  合成视频...")
    FrameExporter.create_video(FRAMES_DIR, output_video, fps)
    
    print(f"\n{'='*60}")
    print(f"✅ 视频导出成功!")
    print(f"📹 文件位置: {output_video}")
    print(f"   时长: {duration}秒")
    print(f"   帧率: {fps} FPS")
    print(f"   总帧数: {total_frames}")
    print('='*60 + "\n")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python export_video.py <动画名> [选项]")
        print("\n必需参数:")
        print("  <动画名>    动画文件名（不含.json）")
        print("\n可选参数:")
        print("  --angle <度数>    相机方位角 (默认: 90)")
        print("                    0=后面, 90=右侧, 180=前面, 270=左侧")
        print("  --mode <模式>     渲染模式 (默认: solid)")
        print("                    solid, wireframe, transparent, wireframe_transparent")
        print("  --fps <帧率>      视频帧率 (默认: 30)")
        print("  --duration <秒>   导出时长 (默认: 0=完整动画)")
        print("\n示例:")
        print("  python export_video.py walk_cycle")
        print("  python export_video.py elk_performance --angle 180 --fps 60")
        print("  python export_video.py tail_wag --mode wireframe_transparent")
        print("  python export_video.py walk_cycle --duration 4")
        sys.exit(1)
    
    anim_name = sys.argv[1]
    
    # 解析参数
    view_angle = 90
    render_mode = 'transparent'
    fps = 30
    duration = 0
    
    try:
        if '--angle' in sys.argv:
            idx = sys.argv.index('--angle')
            if idx + 1 < len(sys.argv):
                view_angle = int(sys.argv[idx + 1])
        
        if '--mode' in sys.argv:
            idx = sys.argv.index('--mode')
            if idx + 1 < len(sys.argv):
                mode = sys.argv[idx + 1]
                if mode in ['solid', 'wireframe', 'transparent', 'wireframe_transparent']:
                    render_mode = mode
        
        if '--fps' in sys.argv:
            idx = sys.argv.index('--fps')
            if idx + 1 < len(sys.argv):
                fps = int(sys.argv[idx + 1])
        
        if '--duration' in sys.argv:
            idx = sys.argv.index('--duration')
            if idx + 1 < len(sys.argv):
                duration = float(sys.argv[idx + 1])
    except:
        print("⚠️ 参数解析错误，使用默认值")
    
    export_animation_video(anim_name, view_angle, render_mode, fps, duration)