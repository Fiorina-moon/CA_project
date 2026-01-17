"""
重新计算权重 - 使用区域分割版
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) 

from src.config import ELK_OBJ_PATH, SKELETON_JSON_PATH, WEIGHTS_DIR
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.skinning.weight_calculator import WeightCalculator
from src.utils.file_io import save_weights_npz
import numpy as np


def recompute_weights(max_influences=4):
    """使用区域分割算法重新计算权重"""
    print("="*60)
    print("重新计算蒙皮权重 (区域分割版)")
    print("="*60)
    
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
    print("\n[2/4] 计算权重...")
    calculator = WeightCalculator(max_influences=max_influences)
    weights = calculator.compute_weights(mesh, skeleton)
    
    # 备份旧文件
    print("\n[3/4] 备份旧权重文件...")
    old_weights_path = WEIGHTS_DIR / "elk_weights.npz"
    if old_weights_path.exists():
        backup_path = WEIGHTS_DIR / "elk_weights_backup.npz"
        import shutil
        shutil.copy(old_weights_path, backup_path)
        print(f"  ✓ 已备份到: {backup_path}")
    
    # 保存新权重
    print("\n[4/4] 保存新权重...")
    save_weights_npz(weights, old_weights_path)
    print(f"  ✓ 已保存到: {old_weights_path}")
    
    print("\n" + "="*60)
    print("✓ 权重重新计算完成！")
    print("="*60)


if __name__ == "__main__":
    recompute_weights(max_influences=4)
