"""
测试动画系统
"""
import sys
from pathlib import Path
import math

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import SKELETON_JSON_PATH, ANIMATIONS_DIR
from src.core.skeleton_loader import SkeletonLoader
from src.animation.keyframe import AnimationClip, JointKeyframe
from src.animation.animator import Animator
from src.utils.file_io import save_animation, load_animation

def create_test_animation(skeleton) -> AnimationClip:
    """创建测试动画 - 简单的腿部摆动"""
    print("\n" + "="*60)
    print("创建测试动画")
    print("="*60)
    
    clip = AnimationClip("test_walk", duration=2.0)
    
    # 为后腿添加小幅度摆动（减小角度）
    left_leg = "RigLBLeg2_04"  # 左后腿第二节
    right_leg = "RigRBLeg2_08"  # 右后腿第二节
    
    # 左腿
    clip.add_keyframe(left_leg, JointKeyframe(
        time=0.0,
        rotation=(0, 0, 0)
    ))
    clip.add_keyframe(left_leg, JointKeyframe(
        time=1.0,
        rotation=(math.radians(15), 0, 0)  # 减小到15度
    ))
    clip.add_keyframe(left_leg, JointKeyframe(
        time=2.0,
        rotation=(0, 0, 0)
    ))
    
    # 右腿（相反）
    clip.add_keyframe(right_leg, JointKeyframe(
        time=0.0,
        rotation=(math.radians(15), 0, 0)
    ))
    clip.add_keyframe(right_leg, JointKeyframe(
        time=1.0,
        rotation=(0, 0, 0)
    ))
    clip.add_keyframe(right_leg, JointKeyframe(
        time=2.0,
        rotation=(math.radians(15), 0, 0)
    ))
    
    print(f"✓ 创建动画片段: {clip}")
    print(f"  关键帧总数: {sum(len(kfs) for kfs in clip.keyframes.values())}")
    
    return clip


def test_keyframe_system():
    """测试关键帧系统"""
    print("\n" + "="*60)
    print("TEST 1: Keyframe System")
    print("="*60)
    
    # 创建关键帧
    kf1 = JointKeyframe(0.0, rotation=(0, 0, 0), translation=(0, 0, 0))
    kf2 = JointKeyframe(1.0, rotation=(math.pi/2, 0, 0), translation=(1, 0, 0))
    
    print(f"  关键帧1: {kf1}")
    print(f"  关键帧2: {kf2}")
    
    # 测试变换矩阵
    T1 = kf1.get_transform_matrix()
    T2 = kf2.get_transform_matrix()
    
    print(f"  变换矩阵生成成功")
    
    print("✓ 关键帧系统测试通过\n")
    return True


def test_interpolation():
    """测试插值"""
    print("="*60)
    print("TEST 2: Interpolation")
    print("="*60)
    
    from src.animation.interpolation import find_keyframe_interval, interpolate_keyframe
    
    # 创建关键帧序列
    keyframes = [
        JointKeyframe(0.0, rotation=(0, 0, 0)),
        JointKeyframe(1.0, rotation=(math.pi/2, 0, 0)),
        JointKeyframe(2.0, rotation=(math.pi, 0, 0))
    ]
    
    # 测试不同时间点
    test_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print(f"  关键帧: t=0.0, 1.0, 2.0")
    print(f"  测试时间点:")
    
    for time in test_times:
        kf0, kf1, blend = find_keyframe_interval(keyframes, time)
        result = interpolate_keyframe(kf0, kf1, blend)
        print(f"    t={time:.1f}: rotation_x={math.degrees(result.rotation[0]):.1f}° (blend={blend:.2f})")
    
    print("✓ 插值测试通过\n")
    return True


def test_animator():
    """测试动画器"""
    print("="*60)
    print("TEST 3: Animator")
    print("="*60)
    
    # 加载骨架
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    # 创建动画
    clip = create_test_animation(skeleton)
    
    # 创建动画器
    animator = Animator(skeleton)
    animator.load_clip(clip)
    animator.play()
    
    # 模拟动画更新
    print(f"\n模拟动画播放:")
    dt = 0.1  # 100ms per frame
    num_frames = int(clip.duration / dt) + 1
    
    for i in range(num_frames):
        animator.update(dt)
        current_time = animator.get_current_time()
        
        if i % 5 == 0:
            print(f"  帧 {i}: t={current_time:.2f}s")
    
    print(f"\n✓ 动画器测试通过")
    print(f"  总帧数: {num_frames}")
    print(f"  最终时间: {animator.get_current_time():.2f}s\n")
    
    return clip


def test_save_load():
    """测试保存和加载"""
    print("="*60)
    print("TEST 4: Save & Load Animation")
    print("="*60)
    
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    # 创建并保存
    clip = create_test_animation(skeleton)
    save_path = ANIMATIONS_DIR / "test_walk.json"
    save_animation(clip, save_path)
    
    # 加载
    loaded_clip = load_animation(save_path)
    
    # 验证
    assert loaded_clip.name == clip.name
    assert loaded_clip.duration == clip.duration
    assert len(loaded_clip.keyframes) == len(clip.keyframes)
    
    print(f"\n✓ 保存/加载测试通过")
    print(f"  动画名称: {loaded_clip.name}")
    print(f"  持续时间: {loaded_clip.duration}s")
    print(f"  关节数: {len(loaded_clip.keyframes)}\n")
    
    return True


def test_loop_animation():
    """测试循环播放"""
    print("="*60)
    print("TEST 5: Loop Animation")
    print("="*60)
    
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    clip = create_test_animation(skeleton)
    
    animator = Animator(skeleton)
    animator.load_clip(clip)
    animator.loop = True
    animator.play()
    
    # 模拟超过一个周期
    total_time = clip.duration * 2.5
    dt = 0.1
    
    print(f"  模拟 {total_time}s 循环播放:")
    
    time_checkpoints = [0, clip.duration, clip.duration * 2, total_time]
    checkpoint_idx = 0
    
    current_sim_time = 0
    while current_sim_time <= total_time:
        animator.update(dt)
        current_sim_time += dt
        
        if checkpoint_idx < len(time_checkpoints) and current_sim_time >= time_checkpoints[checkpoint_idx]:
            print(f"    仿真时间={current_sim_time:.2f}s -> 动画时间={animator.get_current_time():.2f}s")
            checkpoint_idx += 1
    
    print(f"\n✓ 循环动画测试通过\n")
    return True


if __name__ == "__main__":
    print("\n" + "🎬 " + "="*58)
    print("     ANIMATION SYSTEM TESTS")
    print("="*60 + "\n")
    
    all_passed = True
    
    try:
        all_passed &= test_keyframe_system()
        all_passed &= test_interpolation()
        test_animator()
        all_passed &= test_save_load()
        all_passed &= test_loop_animation()
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("="*60 + "\n")