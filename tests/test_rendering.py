"""
测试渲染系统
"""
import sys
from pathlib import Path
import time

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


def test_deformer():
    """测试变形器"""
    print("\n" + "="*60)
    print("TEST 1: Skin Deformer")
    print("="*60)
    
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    weights = load_weights_npz(WEIGHTS_DIR / "elk_weights.npz")
    
    deformer = SkinDeformer(mesh, skeleton, weights)
    
    print(f"\n✓ 变形器创建成功")
    print(f"  绑定顶点: {len(deformer.bind_vertices)}")
    
    # 测试更新
    deformer.update()
    deformed = deformer.get_deformed_vertices()
    
    print(f"  变形顶点: {len(deformed)}")
    print(f"  前3个变形顶点:")
    for i in range(min(3, len(deformed))):
        print(f"    [{i}] {deformed[i]}")
    
    print("\n✓ 变形器测试通过\n")
    return deformer


def test_renderer_static():
    """测试渲染器（静态）"""
    print("="*60)
    print("TEST 2: Renderer (Static)")
    print("="*60)
    
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    # 打印模型和骨架的位置范围
    min_pos, max_pos = mesh.get_bounding_box()
    print(f"\n模型包围盒:")
    print(f"  Min: {min_pos}")
    print(f"  Max: {max_pos}")
    print(f"  中心: {Vector3((min_pos.x + max_pos.x) / 2, (min_pos.y + max_pos.y) / 2, (min_pos.z + max_pos.z) / 2)}")
    
    renderer = Renderer(800, 600, "Test - Static Mesh")
    
    if not renderer.initialize():
        print("✗ 渲染器初始化失败")
        return False
    
    # 根据包围盒设置相机
    center_x = (min_pos.x + max_pos.x) / 2
    center_y = (min_pos.y + max_pos.y) / 2
    center_z = (min_pos.z + max_pos.z) / 2
    
    renderer.camera.target = Vector3(center_x, center_y, center_z)
    
    # 计算合适的距离
    size = max(max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z)
    renderer.camera.distance = size * 2.2
    
    renderer.camera.elevation = 15
    renderer.camera.azimuth = 135
    
    print(f"\n相机设置:")
    print(f"  目标: {renderer.camera.target}")
    print(f"  距离: {renderer.camera.distance:.2f}")
    
    print(f"\n渲染静态场景 (5秒，按ESC退出)...")
    
    start_time = time.time()
    frame_count = 0
    
    while not renderer.should_close() and time.time() - start_time < 5.0:
        renderer.render_frame(mesh, skeleton=skeleton)
        renderer.poll_events()
        frame_count += 1
    
    renderer.cleanup()
    
    fps = frame_count / 5.0
    print(f"✓ 静态渲染测试通过")
    print(f"  总帧数: {frame_count}")
    print(f"  平均FPS: {fps:.1f}\n")
    
    return True


def test_renderer_animated():
    """测试动画渲染 - 先只看骨架"""
    print("="*60)
    print("TEST 3: Renderer (Animated)")
    print("="*60)
    
    # 加载数据
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    animation = load_animation(ANIMATIONS_DIR / "test_walk.json")
    
    # 只创建动画器，不创建deformer
    animator = Animator(skeleton)
    animator.load_clip(animation)
    animator.play()
    
    # 创建渲染器
    renderer = Renderer(800, 600, "Test - Skeleton Animation Only")
    if not renderer.initialize():
        return False
    
    # 根据包围盒设置相机
    min_pos, max_pos = mesh.get_bounding_box()
    center_x = (min_pos.x + max_pos.x) / 2
    center_y = (min_pos.y + max_pos.y) / 2
    center_z = (min_pos.z + max_pos.z) / 2
    
    renderer.camera.target = Vector3(center_x, center_y, center_z)
    size = max(max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z)
    renderer.camera.distance = size * 2.0
    renderer.camera.elevation = 25
    renderer.camera.azimuth = 45
    
    print(f"\n渲染骨架动画 (5秒，观察骨架是否会动)...")
    
    start_time = time.time()
    last_time = start_time
    frame_count = 0
    
    while not renderer.should_close() and time.time() - start_time < 5.0:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # 更新动画
        animator.update(dt)
        
        # 只渲染骨架，不渲染变形的mesh
        renderer.render_frame(mesh, deformer=None, skeleton=skeleton)
        renderer.poll_events()
        
        # 每30帧打印一次动画时间
        if frame_count % 30 == 0:
            print(f"  帧 {frame_count}: 动画时间={animator.get_current_time():.2f}s")
        
        frame_count += 1
    
    renderer.cleanup()
    
    fps = frame_count / 5.0
    print(f"✓ 骨架动画测试")
    print(f"  总帧数: {frame_count}")
    print(f"  平均FPS: {fps:.1f}\n")
    
    return True

def diagnose_alignment():
    """诊断骨架和模型对齐问题"""
    print("\n" + "="*60)
    print("DIAGNOSIS: Skeleton-Mesh Alignment")
    print("="*60)
    
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    # 模型包围盒
    min_pos, max_pos = mesh.get_bounding_box()
    print(f"\n模型包围盒:")
    print(f"  Min: ({min_pos.x:.4f}, {min_pos.y:.4f}, {min_pos.z:.4f})")
    print(f"  Max: ({max_pos.x:.4f}, {max_pos.y:.4f}, {max_pos.z:.4f})")
    
    mesh_center = Vector3(
        (min_pos.x + max_pos.x) / 2,
        (min_pos.y + max_pos.y) / 2,
        (min_pos.z + max_pos.z) / 2
    )
    print(f"  中心: ({mesh_center.x:.4f}, {mesh_center.y:.4f}, {mesh_center.z:.4f})")
    
    # 骨架范围
    joint_positions = [j.head for j in skeleton.joints]
    
    min_x = min(p.x for p in joint_positions)
    max_x = max(p.x for p in joint_positions)
    min_y = min(p.y for p in joint_positions)
    max_y = max(p.y for p in joint_positions)
    min_z = min(p.z for p in joint_positions)
    max_z = max(p.z for p in joint_positions)
    
    print(f"\n骨架包围盒:")
    print(f"  Min: ({min_x:.4f}, {min_y:.4f}, {min_z:.4f})")
    print(f"  Max: ({max_x:.4f}, {max_y:.4f}, {max_z:.4f})")
    
    skeleton_center = Vector3(
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        (min_z + max_z) / 2
    )
    print(f"  中心: ({skeleton_center.x:.4f}, {skeleton_center.y:.4f}, {skeleton_center.z:.4f})")
    
    # 偏移量
    offset = mesh_center - skeleton_center
    print(f"\n偏移量:")
    print(f"  ΔX: {offset.x:.4f}")
    print(f"  ΔY: {offset.y:.4f}")
    print(f"  ΔZ: {offset.z:.4f}")
    
    # 缩放比例
    mesh_size = Vector3(max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z)
    skeleton_size = Vector3(max_x - min_x, max_y - min_y, max_z - min_z)
    
    print(f"\n模型尺寸: ({mesh_size.x:.4f}, {mesh_size.y:.4f}, {mesh_size.z:.4f})")
    print(f"骨架尺寸: ({skeleton_size.x:.4f}, {skeleton_size.y:.4f}, {skeleton_size.z:.4f})")
    
    if skeleton_size.x > 0:
        scale_x = mesh_size.x / skeleton_size.x
    else:
        scale_x = 1.0
    
    if skeleton_size.y > 0:
        scale_y = mesh_size.y / skeleton_size.y
    else:
        scale_y = 1.0
        
    if skeleton_size.z > 0:
        scale_z = mesh_size.z / skeleton_size.z
    else:
        scale_z = 1.0
    
    print(f"\n缩放比例: ({scale_x:.4f}, {scale_y:.4f}, {scale_z:.4f})")
    
    print("\n" + "="*60 + "\n")
    
    return {
        'mesh_center': mesh_center,
        'skeleton_center': skeleton_center,
        'offset': offset,
        'mesh_size': mesh_size,
        'skeleton_size': skeleton_size,
        'scale': (scale_x, scale_y, scale_z)
    }

def test_export_video():
    """测试视频导出"""
    print("="*60)
    print("TEST 4: Video Export")
    print("="*60)
    
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    weights = load_weights_npz(WEIGHTS_DIR / "elk_weights.npz")
    animation = load_animation(ANIMATIONS_DIR / "test_walk.json")
    
    deformer = SkinDeformer(mesh, skeleton, weights)
    animator = Animator(skeleton)
    animator.load_clip(animation)
    animator.play()
    
    renderer = Renderer(800, 600, "Export Video")
    if not renderer.initialize():
        return False
    
    # 🔧 相机设置 - 侧面视角
    min_pos, max_pos = mesh.get_bounding_box()
    center_x = (min_pos.x + max_pos.x) / 2
    center_y = (min_pos.y + max_pos.y) / 2
    center_z = (min_pos.z + max_pos.z) / 2
    
    renderer.camera.target = Vector3(center_x, center_y, center_z)
    size = max(max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z)
    renderer.camera.distance = size * 2.5
    renderer.camera.elevation = 10      # 降低仰角看侧面
    renderer.camera.azimuth = 90         # 正侧面视角（0度或90度）
    
    exporter = FrameExporter(800, 600)
    
    fps = 30
    duration = 4.0
    total_frames = int(duration * fps)
    dt = 1.0 / fps
    
    print(f"\n导出设置:")
    print(f"  帧率: {fps} FPS")
    print(f"  时长: {duration}s")
    print(f"  总帧数: {total_frames}")
    print(f"  相机: 侧面视角 (azimuth={renderer.camera.azimuth}°)")
    
    # 清空帧目录
    for old_frame in FRAMES_DIR.glob("frame_*.png"):
        old_frame.unlink()
    
    print(f"\n开始渲染帧...")
    
    for frame_idx in range(total_frames):
        if (frame_idx + 1) % 30 == 0:
            print(f"  进度: {frame_idx + 1}/{total_frames}")
        
        animator.update(dt)
        deformer.update()
        
        renderer.render_frame(mesh, deformer, skeleton)
        
        image = exporter.capture_frame()
        frame_path = FRAMES_DIR / f"frame_{frame_idx:04d}.png"
        exporter.save_frame(image, frame_path)
        
        renderer.poll_events()
    
    renderer.cleanup()
    
    print(f"✓ 帧渲染完成: {total_frames} 帧")
    
    output_video = VIDEOS_DIR / "elk_animation.mp4"
    FrameExporter.create_video(FRAMES_DIR, output_video, fps)
    
    print(f"\n✓ 视频导出测试通过\n")
    return True

if __name__ == "__main__":
    print("\n" + "🎥 " + "="*58)
    print("     RENDERING SYSTEM TESTS")
    print("="*60 + "\n")
    
    all_passed = True
    
    try:
        # 诊断
        diagnose_alignment()
        
        # 完整测试流程
        test_deformer()
        all_passed &= test_renderer_static()
        all_passed &= test_renderer_animated()  # ✅ 启用动画测试
        all_passed &= test_export_video()       # ✅ 启用视频导出
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print(f"\n📹 最终视频: {VIDEOS_DIR / 'elk_animation.mp4'}")
    else:
        print("✗ SOME TESTS FAILED!")
    print("="*60 + "\n")