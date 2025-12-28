"""
快速测试新动画
用法: python quick_test.py tail_wag / head_nod / walk_cycle
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import ELK_OBJ_PATH, SKELETON_JSON_PATH, WEIGHTS_DIR, ANIMATIONS_DIR
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.skinning.deformer import SkinDeformer
from src.animation.animator import Animator
from src.rendering.renderer import Renderer
from src.utils.file_io import load_weights_npz, load_animation
from src.utils.math_utils import Vector3
import time


def test_animation(animation_name):
    """测试指定的动画文件"""
    print(f"\n{'='*60}")
    print(f"测试动画: {animation_name}")
    print('='*60)
    
    # 加载数据
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    weights = load_weights_npz(WEIGHTS_DIR / "elk_weights.npz")
    
    # 构建动画文件路径：data/animations/xxx.json
    if not animation_name.endswith('.json'):
        animation_name += '.json'
    
    anim_path = ANIMATIONS_DIR / animation_name
    
    if not anim_path.exists():
        print(f"❌ 文件不存在: {anim_path}")
        print(f"\n请确保文件在: data/animations/{animation_name}")
        return
    
    # 使用load_animation函数加载
    animation = load_animation(anim_path)
    
    print(f"\n动画信息:")
    print(f"  路径: {anim_path}")
    print(f"  名称: {animation.name}")
    print(f"  时长: {animation.duration}秒")
    if hasattr(animation, 'keyframes'):
        print(f"  关键帧关节数: {len(animation.keyframes)}")
        print(f"  关键帧关节列表: {list(animation.keyframes.keys())}")
    else:
        print(f"  关键帧关节数: 未知")
    
    # 创建动画系统
    deformer = SkinDeformer(mesh, skeleton, weights)
    animator = Animator(skeleton)
    animator.load_clip(animation)
    animator.play()
    
    # 创建渲染器
    renderer = Renderer(800, 600, f"Test - {animation.name}")
    if not renderer.initialize():
        return
    
    # 根据动画类型设置相机
    min_pos, max_pos = mesh.get_bounding_box()
    center = Vector3(
        (min_pos.x + max_pos.x) / 2,
        (min_pos.y + max_pos.y) / 2,
        (min_pos.z + max_pos.z) / 2
    )
    size = max(max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z)
    
    renderer.camera.target = center
    renderer.camera.distance = size * 2.5
    
    # 根据动画类型调整视角
    if 'tail' in animation.name.lower():
        print("\n📷 相机设置: 后视角 (看尾巴)")
        renderer.camera.azimuth = 180
        renderer.camera.elevation = 15
    elif 'head' in animation.name.lower() or 'nod' in animation.name.lower():
        print("\n📷 相机设置: 侧视角 (看头部)")
        renderer.camera.azimuth = 90
        renderer.camera.elevation = 5
    else:
        print("\n📷 相机设置: 斜侧视角 (看整体)")
        renderer.camera.azimuth = 180
        renderer.camera.elevation = 15
    
    print(f"   实际相机参数: azimuth={renderer.camera.azimuth}°, elevation={renderer.camera.elevation}°")
    
    print(f"\n🎬 开始渲染 (按ESC退出)...")
    print(f"   动画会循环播放 {animation.duration}秒")
    
    start_time = time.time()
    last_time = start_time
    frame_count = 0
    
    while not renderer.should_close():
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # 更新动画和变形
        animator.update(dt)
        deformer.update()
        
        # 渲染
        renderer.render_frame(mesh, deformer, skeleton)
        renderer.poll_events()
        
        # 每秒打印一次状态
        if frame_count % 30 == 0:
            print(f"  时间: {animator.get_current_time():.2f}s / {animation.duration:.2f}s")
        
        frame_count += 1
    
    renderer.cleanup()
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n✓ 测试完成")
    print(f"  总帧数: {frame_count}")
    print(f"  平均FPS: {fps:.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python quick_test.py <动画名>")
        print("\n可用动画:")
        print("  tail_wag     - 尾巴摆动 (推荐先测试)")
        print("  head_nod     - 头部点头")
        print("  walk_cycle   - 完整行走")
        print("\n示例:")
        print("  python quick_test.py tail_wag")
        print("  python quick_test.py head_nod.json")
        print("\n动画文件路径: data/animations/")
        sys.exit(1)
    
    anim_name = sys.argv[1]
    test_animation(anim_name)