"""
改进的权重诊断脚本 - 修正腿部区域判断
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
from src.utils.file_io import load_weights_npz


def diagnose_weights_improved():
    """改进的权重诊断"""
    print("="*60)
    print("权重诊断 (改进版)")
    print("="*60)
    
    # 加载数据
    mesh = OBJLoader.load(ELK_OBJ_PATH)
    skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
    weights = load_weights_npz(WEIGHTS_DIR / "elk_weights.npz")
    
    vertices_array = np.array([[v.x, v.y, v.z] for v in mesh.vertices])
    
    print(f"\n基本信息:")
    print(f"  顶点数: {mesh.get_vertex_count()}")
    print(f"  骨骼数: {skeleton.get_bone_count()}")
    
    # 找到腿部骨骼
    leg_bones = []
    for i, bone in enumerate(skeleton.bones):
        if 'leg' in bone.name.lower():
            leg_bones.append(i)
    
    print(f"  腿部骨骼数: {len(leg_bones)}")
    
    # 分析顶点Y坐标分布
    y_coords = vertices_array[:, 1]
    print(f"\nY坐标统计:")
    print(f"  最小: {y_coords.min():.3f}")
    print(f"  25%分位: {np.percentile(y_coords, 25):.3f}")
    print(f"  中位数: {np.median(y_coords):.3f}")
    print(f"  75%分位: {np.percentile(y_coords, 75):.3f}")
    print(f"  最大: {y_coords.max():.3f}")
    
    # 分析骨骼位置
    print(f"\n骨骼Y坐标分布:")
    leg_bone_ys = []
    for bone_idx in leg_bones[:4]:  # 看前4根腿骨
        bone = skeleton.bones[bone_idx]
        start_y = bone.start_joint.head.y
        end_y = bone.end_joint.head.y
        avg_y = (start_y + end_y) / 2
        leg_bone_ys.append(avg_y)
        print(f"  {bone.name[:40]:40s} Y={avg_y:.3f}")
    
    if leg_bone_ys:
        avg_leg_bone_y = np.mean(leg_bone_ys)
        print(f"  平均腿部骨骼Y坐标: {avg_leg_bone_y:.3f}")
    
    # 修正：腿部区域应该是Y坐标较低的部分
    print(f"\n{'='*60}")
    print("修正后的腿部区域分析")
    print("="*60)
    
    # 方法1: 使用Y坐标的下25%作为腿部区域
    y_25th = np.percentile(y_coords, 25)
    leg_region_v1 = y_coords < y_25th
    
    print(f"\n方法1: Y < 25%分位 ({y_25th:.3f})")
    print(f"  腿部区域顶点: {leg_region_v1.sum()}")
    
    if leg_region_v1.sum() > 0 and leg_bones:
        leg_weights = weights[leg_region_v1][:, leg_bones]
        total_leg_influence = leg_weights.sum(axis=1).mean()
        print(f"  腿部骨骼平均影响: {total_leg_influence:.1%}")
        
        # 检查这些顶点主要受哪些骨骼控制
        all_influences = weights[leg_region_v1].sum(axis=0)
        top_bone_indices = np.argsort(all_influences)[-5:][::-1]
        
        print(f"\n  主要影响骨骼 (Top 5):")
        for rank, bone_idx in enumerate(top_bone_indices, 1):
            bone = skeleton.bones[bone_idx]
            influence = all_influences[bone_idx]
            is_leg = "✓" if bone_idx in leg_bones else "✗"
            print(f"    {rank}. [{is_leg}] {bone.name[:45]:45s} = {influence:.1f}")
    
    # 方法2: 基于腿部骨骼的平均Y坐标
    if leg_bone_ys:
        y_threshold = avg_leg_bone_y + 0.2  # 稍微高一点
        leg_region_v2 = y_coords < y_threshold
        
        print(f"\n方法2: Y < 腿骨Y+0.2 ({y_threshold:.3f})")
        print(f"  腿部区域顶点: {leg_region_v2.sum()}")
        
        if leg_region_v2.sum() > 0:
            leg_weights = weights[leg_region_v2][:, leg_bones]
            total_leg_influence = leg_weights.sum(axis=1).mean()
            print(f"  腿部骨骼平均影响: {total_leg_influence:.1%}")
    
    # 分析具体的腿部顶点
    print(f"\n{'='*60}")
    print("具体顶点分析")
    print("="*60)
    
    # 找到真正的腿部顶点（Y坐标很低的）
    lowest_y_indices = np.argsort(y_coords)[:20]  # Y最小的20个顶点
    
    print(f"\nY坐标最低的20个顶点 (应该是腿部/脚部):")
    for i, v_idx in enumerate(lowest_y_indices[:5]):
        y = y_coords[v_idx]
        # 找到对这个顶点影响最大的骨骼
        top_bone = np.argmax(weights[v_idx])
        top_weight = weights[v_idx, top_bone]
        bone = skeleton.bones[top_bone]
        is_leg = "✓" if top_bone in leg_bones else "✗"
        
        print(f"  顶点{v_idx:4d}: Y={y:.3f}, 主要骨骼=[{is_leg}] {bone.name[:35]:35s} ({top_weight:.3f})")
    
    # 检查权重值分布
    print(f"\n{'='*60}")
    print("权重值分布分析")
    print("="*60)
    
    for bone_idx in leg_bones[:4]:
        bone = skeleton.bones[bone_idx]
        bone_weights = weights[:, bone_idx]
        
        # 统计这个骨骼的权重分布
        affected = (bone_weights > 0.01).sum()
        max_weight = bone_weights.max()
        avg_weight = bone_weights[bone_weights > 0.01].mean() if affected > 0 else 0
        
        # 有多少顶点是主要由这个骨骼控制的（权重>0.5）
        dominant = (bone_weights > 0.5).sum()
        
        print(f"\n  {bone.name[:40]}")
        print(f"    影响顶点: {affected}")
        print(f"    最大权重: {max_weight:.3f}")
        print(f"    平均权重: {avg_weight:.3f}")
        print(f"    主导顶点 (>0.5): {dominant}")
    
    # 诊断建议
    print(f"\n{'='*60}")
    print("诊断建议")
    print("="*60)
    
    issues = []
    
    # 检查方法1的结果
    if leg_region_v1.sum() > 0 and leg_bones:
        leg_weights = weights[leg_region_v1][:, leg_bones]
        total_leg_influence = leg_weights.sum(axis=1).mean()
        
        if total_leg_influence < 0.3:
            issues.append("腿部顶点主要不受腿部骨骼控制")
        
        # 检查有多少腿部骨骼的最大权重很低
        low_weight_bones = 0
        for bone_idx in leg_bones:
            if weights[:, bone_idx].max() < 0.4:
                low_weight_bones += 1
        
        if low_weight_bones > len(leg_bones) * 0.3:
            issues.append(f"{low_weight_bones}根腿部骨骼的最大权重<0.4（影响力太弱）")
    
    # 检查距离计算
    print(f"\n检查骨骼距离计算:")
    
    # 随机选一个低Y坐标的顶点
    if len(lowest_y_indices) > 0:
        test_v_idx = lowest_y_indices[0]
        test_v = vertices_array[test_v_idx]
        
        print(f"\n  测试顶点 {test_v_idx}: ({test_v[0]:.3f}, {test_v[1]:.3f}, {test_v[2]:.3f})")
        
        # 计算到前4根腿骨的距离
        from utils.geometry import point_to_segment_distance
        
        for bone_idx in leg_bones[:4]:
            bone = skeleton.bones[bone_idx]
            dist = point_to_segment_distance(
                mesh.vertices[test_v_idx],
                bone.start_joint.head,
                bone.end_joint.head
            )
            weight = weights[test_v_idx, bone_idx]
            print(f"    到 {bone.name[:35]:35s}: 距离={dist:.3f}, 权重={weight:.3f}")
    
    if issues:
        print(f"\n发现的问题:")
        for issue in issues:
            print(f"  ❌ {issue}")
        
        print(f"\n建议修复方案:")
        print(f"  1. 大幅增加影响半径 (从0.45 → 0.6)")
        print(f"  2. 调整权重公式，让近距离骨骼权重更大")
        print(f"  3. 考虑使用更好的算法（如Heat Diffusion）")
    else:
        print(f"\n✅ 权重分配基本正常")


if __name__ == "__main__":
    diagnose_weights_improved()