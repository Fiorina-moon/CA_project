"""
测试核心加载器
"""
import sys
from pathlib import Path

# 添加src到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import ELK_OBJ_PATH, SKELETON_JSON_PATH 
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.utils.math_utils import Vector3
from src.utils.geometry import point_to_segment_distance


def test_mesh_loader():
    """测试Mesh加载器"""
    print("\n" + "="*60)
    print("TEST 1: Mesh Loader")
    print("="*60)
    
    # 检查文件是否存在
    if not ELK_OBJ_PATH.exists():
        print(f"✗ File not found: {ELK_OBJ_PATH}")
        return False
    
    # 加载mesh
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    
    # 验证数据
    assert mesh.get_vertex_count() > 0, "No vertices loaded"
    assert mesh.get_face_count() > 0, "No faces loaded"
    
    # 显示包围盒
    min_pos, max_pos = mesh.get_bounding_box()
    print(f"  Bounding box:")
    print(f"    Min: {min_pos}")
    print(f"    Max: {max_pos}")
    
    # 显示前5个顶点
    print(f"  First 5 vertices:")
    for i in range(min(5, len(mesh.vertices))):
        print(f"    [{i}] {mesh.vertices[i]}")
    
    # 显示前3个面
    print(f"  First 3 faces:")
    for i in range(min(3, len(mesh.faces))):
        print(f"    [{i}] {mesh.faces[i]}")
    
    print("✓ Mesh loader test passed!\n")
    return True


def test_skeleton_loader():
    """测试Skeleton加载器"""
    print("="*60)
    print("TEST 2: Skeleton Loader")
    print("="*60)
    
    # 检查文件是否存在
    if not SKELETON_JSON_PATH.exists():
        print(f"✗ File not found: {SKELETON_JSON_PATH}")
        return False
    
    # 加载skeleton
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    # 验证数据
    assert skeleton.get_joint_count() > 0, "No joints loaded"
    assert skeleton.root_joint is not None, "No root joint found"
    
    # 显示层级结构
    print(f"\n  Skeleton hierarchy:")
    print_hierarchy(skeleton.root_joint, indent=2)
    
    # 显示骨骼信息
    print(f"\n  Bones ({skeleton.get_bone_count()}):")
    for i in range(min(5, len(skeleton.bones))):
        bone = skeleton.bones[i]
        print(f"    [{i}] {bone.name}")
        print(f"        Length: {bone.get_length():.4f}")
        print(f"        Direction: {bone.get_direction()}")
    
    print("\n✓ Skeleton loader test passed!\n")
    return True


def print_hierarchy(joint, indent=0):
    """递归打印骨架层级"""
    prefix = " " * indent
    print(f"{prefix}├─ {joint.name} (index={joint.index})")
    print(f"{prefix}│  head: {joint.head}")
    print(f"{prefix}│  tail: {joint.tail}")
    
    for child in joint.children:
        print_hierarchy(child, indent + 3)


def test_geometry_utils():
    """测试几何工具"""
    print("="*60)
    print("TEST 3: Geometry Utils")
    print("="*60)
    
    # 测试点到线段距离
    point = Vector3(0, 1, 0)
    seg_start = Vector3(-1, 0, 0)
    seg_end = Vector3(1, 0, 0)
    
    distance = point_to_segment_distance(point, seg_start, seg_end)
    print(f"  Point: {point}")
    print(f"  Segment: {seg_start} -> {seg_end}")
    print(f"  Distance: {distance:.4f}")
    print(f"  Expected: 1.0000")
    
    assert abs(distance - 1.0) < 1e-4, "Distance calculation error"
    
    # 测试点在线段延长线上的情况
    point2 = Vector3(2, 0, 0)
    distance2 = point_to_segment_distance(point2, seg_start, seg_end)
    print(f"\n  Point: {point2}")
    print(f"  Distance: {distance2:.4f}")
    print(f"  Expected: 1.0000")
    
    assert abs(distance2 - 1.0) < 1e-4, "Distance calculation error (beyond segment)"
    
    print("\n✓ Geometry utils test passed!\n")
    return True


def test_data_consistency():
    """测试数据一致性"""
    print("="*60)
    print("TEST 4: Data Consistency")
    print("="*60)
    
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    print(f"  Mesh vertices: {mesh.get_vertex_count()}")
    print(f"  Skeleton joints: {skeleton.get_joint_count()}")
    print(f"  Skeleton bones: {skeleton.get_bone_count()}")
    
    # 检查骨架是否在模型范围内
    min_pos, max_pos = mesh.get_bounding_box()
    
    joints_outside = 0
    for joint in skeleton.joints:
        head = joint.head
        if not (min_pos.x <= head.x <= max_pos.x and
                min_pos.y <= head.y <= max_pos.y and
                min_pos.z <= head.z <= max_pos.z):
            joints_outside += 1
    
    print(f"  Joints outside bounding box: {joints_outside}/{skeleton.get_joint_count()}")
    
    if joints_outside > skeleton.get_joint_count() * 0.5:
        print("  ⚠ Warning: Many joints are outside mesh bounding box")
    
    print("\n✓ Data consistency test passed!\n")
    return True


if __name__ == "__main__":
    print("\n" + "🔧 " + "="*58)
    print("     SKELETAL ANIMATION - CORE MODULE TESTS")
    print("="*60 + "\n")
    
    all_passed = True
    
    # 运行所有测试
    try:
        all_passed &= test_mesh_loader()
        all_passed &= test_skeleton_loader()
        all_passed &= test_geometry_utils()
        all_passed &= test_data_consistency()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 总结
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED!")
    print("="*60 + "\n")