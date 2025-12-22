"""
测试权重计算器
"""
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import ELK_OBJ_PATH, SKELETON_JSON_PATH, WEIGHTS_DIR
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.skinning.weight_calculator import WeightCalculator
from src.utils.file_io import save_weights_npz, load_weights_npz


def test_weight_calculation():
    """测试权重计算"""
    print("\n" + "="*60)
    print("TEST: Weight Calculation (Bilinear)")
    print("="*60)
    
    # 加载数据
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    # 创建权重计算器
    calculator = WeightCalculator()
    
    # 计算权重
    weights = calculator.compute_weights_bilinear(mesh, skeleton)
    
    # 验证形状
    expected_shape = (mesh.get_vertex_count(), skeleton.get_bone_count())
    assert weights.shape == expected_shape, f"权重形状错误: {weights.shape} != {expected_shape}"
    
    # 保存权重
    output_path = WEIGHTS_DIR / "elk_weights.npz"
    save_weights_npz(weights, output_path)
    
    # 测试加载
    loaded_weights = load_weights_npz(output_path)
    assert np.allclose(weights, loaded_weights), "加载的权重与原始权重不匹配"
    
    # 显示权重统计
    print(f"\n权重统计:")
    print(f"  形状: {weights.shape}")
    print(f"  最小值: {weights.min():.6f}")
    print(f"  最大值: {weights.max():.6f}")
    print(f"  平均值: {weights.mean():.6f}")
    
    # 显示每个顶点的权重分布示例
    print(f"\n前5个顶点的权重分布:")
    for i in range(min(5, weights.shape[0])):
        non_zero_indices = np.where(weights[i] > 1e-6)[0]
        print(f"  顶点 {i}:")
        for bone_idx in non_zero_indices:
            print(f"    骨骼 {bone_idx}: {weights[i, bone_idx]:.4f}")
    
    print("\n✓ 权重计算测试通过!\n")
    return True


def test_nearest_method():
    """测试最近邻法（对比）"""
    print("="*60)
    print("TEST: Weight Calculation (Nearest)")
    print("="*60)
    
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    
    calculator = WeightCalculator()
    weights = calculator.compute_weights_nearest(mesh, skeleton)
    
    # 验证最近邻法特性：每行只有一个1
    rows_with_single_one = (weights == 1.0).sum(axis=1)
    assert np.all(rows_with_single_one == 1), "最近邻法每行应该只有一个1"
    
    print(f"✓ 最近邻法测试通过")
    print(f"  每个顶点只受1个骨骼影响\n")
    return True


def visualize_weight_distribution(weights: np.ndarray):
    """可视化权重分布"""
    print("="*60)
    print("Weight Distribution Analysis")
    print("="*60)
    
    # 每个顶点受多少骨骼影响
    bones_per_vertex = (weights > 1e-6).sum(axis=1)
    
    print(f"\n每个顶点影响的骨骼数量:")
    unique, counts = np.unique(bones_per_vertex, return_counts=True)
    for num_bones, count in zip(unique, counts):
        percentage = count / len(bones_per_vertex) * 100
        print(f"  {int(num_bones)} 个骨骼: {count} 顶点 ({percentage:.2f}%)")
    
    # 每个骨骼影响多少顶点
    vertices_per_bone = (weights > 1e-6).sum(axis=0)
    
    print(f"\n每个骨骼影响的顶点数量:")
    print(f"  最小: {vertices_per_bone.min():.0f}")
    print(f"  最大: {vertices_per_bone.max():.0f}")
    print(f"  平均: {vertices_per_bone.mean():.0f}")
    
    # 找出影响最大的骨骼
    top_5_bones = np.argsort(vertices_per_bone)[-5:][::-1]
    print(f"\n影响顶点最多的5个骨骼:")
    for rank, bone_idx in enumerate(top_5_bones, 1):
        print(f"  {rank}. 骨骼 {bone_idx}: {vertices_per_bone[bone_idx]:.0f} 顶点")
    
    print()


if __name__ == "__main__":
    print("\n" + "🔧 " + "="*58)
    print("     WEIGHT CALCULATION TESTS")
    print("="*60 + "\n")
    
    all_passed = True
    
    try:
        # 测试双线性插值法
        all_passed &= test_weight_calculation()
        
        # 测试最近邻法
        all_passed &= test_nearest_method()
        
        # 加载并分析权重分布
        weights = load_weights_npz(WEIGHTS_DIR / "elk_weights.npz")
        visualize_weight_distribution(weights)
        
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