import json
import argparse
from pathlib import Path
from copy import deepcopy

# 默认的骨骼变换值
DEFAULT_TRANSFORM = {
    "rotation": [0, 0, 0],
    "translation": [0, 0, 0],
    "scale": [1, 1, 1]
}

def load_animation(filepath):
    """加载单个动画JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_keyframe_at_time(keyframes, time, default=None):
    """获取指定时间点的关键帧值（精确匹配或最近的前一帧）"""
    if not keyframes:
        return default
    
    # 按时间排序
    sorted_kfs = sorted(keyframes, key=lambda x: x["time"])
    
    # 找到 <= time 的最后一帧
    result = None
    for kf in sorted_kfs:
        if kf["time"] <= time:
            result = kf
        else:
            break
    
    return result if result else default

def get_last_keyframe(keyframes):
    """获取动画的最后一帧"""
    if not keyframes:
        return None
    return max(keyframes, key=lambda x: x["time"])

def get_first_keyframe(keyframes):
    """获取动画的第一帧"""
    if not keyframes:
        return None
    return min(keyframes, key=lambda x: x["time"])

def create_keyframe(time, rotation=None, translation=None, scale=None):
    """创建一个关键帧"""
    return {
        "time": round(time, 4),
        "rotation": rotation if rotation else [0, 0, 0],
        "translation": translation if translation else [0, 0, 0],
        "scale": scale if scale else [1, 1, 1]
    }

def merge_animations(animation_files, transition_time=0.3, output_name="merged_animation"):
    """
    将多个动画文件拼接成一个长动画
    
    修正逻辑：
    - 维护每个骨骼的"当前状态"
    - 动画未覆盖的骨骼保持之前的状态
    - 在动画切换点插入必要的保持帧
    """
    if not animation_files:
        raise ValueError("至少需要一个动画文件")
    
    # 加载所有动画
    animations = []
    for filepath in animation_files:
        anim = load_animation(filepath)
        animations.append(anim)
        print(f"已加载: {filepath} (时长: {anim['duration']}s)")
    
    # 第一遍：收集所有骨骼名称
    all_bones = set()
    for anim in animations:
        all_bones.update(anim["keyframes"].keys())
    
    print(f"共发现 {len(all_bones)} 个骨骼")
    
    # 初始化合并后的动画
    merged = {
        "name": output_name,
        "duration": 0,
        "keyframes": {bone: [] for bone in all_bones}
    }
    
    # 维护每个骨骼的当前状态
    bone_states = {
        bone: deepcopy(DEFAULT_TRANSFORM) for bone in all_bones
    }
    
    current_time = 0.0
    
    for idx, anim in enumerate(animations):
        anim_duration = anim["duration"]
        anim_start_time = current_time
        anim_end_time = current_time + anim_duration
        
        print(f"\n处理动画 {idx + 1}: {anim.get('name', 'unnamed')} @ {anim_start_time:.2f}s - {anim_end_time:.2f}s")
        
        # 记录这个动画涉及的骨骼
        animated_bones = set(anim["keyframes"].keys())
        static_bones = all_bones - animated_bones
        
        # 对于这个动画中没有涉及的骨骼，在开始和结束时插入保持帧
        for bone in static_bones:
            state = bone_states[bone]
            
            # 在动画开始时插入当前状态（保持帧）
            hold_frame_start = create_keyframe(
                anim_start_time,
                state["rotation"],
                state["translation"],
                state["scale"]
            )
            merged["keyframes"][bone].append(hold_frame_start)
            
            # 在动画结束时也插入保持帧（确保整个时间段内保持不变）
            hold_frame_end = create_keyframe(
                anim_end_time,
                state["rotation"],
                state["translation"],
                state["scale"]
            )
            merged["keyframes"][bone].append(hold_frame_end)
        
        # 对于这个动画涉及的骨骼
        for bone in animated_bones:
            keyframes = anim["keyframes"][bone]
            
            if not keyframes:
                continue
            
            first_kf = get_first_keyframe(keyframes)
            last_kf = get_last_keyframe(keyframes)
            
            # 检查动画是否从 time=0 开始
            # 如果不是，需要在动画开始时插入一个从当前状态过渡的帧
            if first_kf["time"] > 0.001:
                # 在动画开始时插入当前状态
                state = bone_states[bone]
                pre_frame = create_keyframe(
                    anim_start_time,
                    state["rotation"],
                    state["translation"],
                    state["scale"]
                )
                merged["keyframes"][bone].append(pre_frame)
            
            # 添加动画的所有关键帧（时间偏移）
            for kf in keyframes:
                new_kf = deepcopy(kf)
                new_kf["time"] = round(kf["time"] + current_time, 4)
                merged["keyframes"][bone].append(new_kf)
            
            # 更新骨骼状态为动画结束时的状态
            bone_states[bone] = {
                "rotation": deepcopy(last_kf["rotation"]),
                "translation": deepcopy(last_kf["translation"]),
                "scale": deepcopy(last_kf["scale"])
            }
        
        # 更新时间偏移
        current_time = anim_end_time
        
        # 添加过渡时间（如果不是最后一个动画）
        if idx < len(animations) - 1 and transition_time > 0:
            # 在过渡期间，所有骨骼保持当前状态
            transition_end = current_time + transition_time
            
            for bone in all_bones:
                state = bone_states[bone]
                hold_frame = create_keyframe(
                    transition_end,
                    state["rotation"],
                    state["translation"],
                    state["scale"]
                )
                merged["keyframes"][bone].append(hold_frame)
            
            current_time = transition_end
    
    merged["duration"] = round(current_time, 4)
    
    # 按时间排序并去重
    for bone_name in merged["keyframes"]:
        # 去重（相同时间点只保留一个）
        seen_times = {}
        unique_kfs = []
        
        for kf in sorted(merged["keyframes"][bone_name], key=lambda x: x["time"]):
            t = round(kf["time"], 4)
            if t not in seen_times:
                seen_times[t] = True
                unique_kfs.append(kf)
        
        merged["keyframes"][bone_name] = unique_kfs
    
    # 移除空的骨骼轨道
    merged["keyframes"] = {
        k: v for k, v in merged["keyframes"].items() if v
    }
    
    return merged

def merge_with_smooth_transition(animation_files, transition_time=0.5, transition_steps=5, output_name="smooth_merged"):
    """
    带平滑过渡的动画合并
    在动画切换时，对所有骨骼进行插值过渡
    """
    if not animation_files:
        raise ValueError("至少需要一个动画文件")
    
    animations = [load_animation(f) for f in animation_files]
    
    # 收集所有骨骼
    all_bones = set()
    for anim in animations:
        all_bones.update(anim["keyframes"].keys())
    
    merged = {
        "name": output_name,
        "duration": 0,
        "keyframes": {bone: [] for bone in all_bones}
    }
    
    # 维护骨骼状态
    bone_states = {bone: deepcopy(DEFAULT_TRANSFORM) for bone in all_bones}
    
    current_time = 0.0
    
    for idx, anim in enumerate(animations):
        anim_duration = anim["duration"]
        animated_bones = set(anim["keyframes"].keys())
        
        # 计算下一个动画每个骨骼的起始状态（用于过渡）
        next_anim_starts = {}
        for bone in all_bones:
            if bone in anim["keyframes"] and anim["keyframes"][bone]:
                first_kf = get_first_keyframe(anim["keyframes"][bone])
                next_anim_starts[bone] = {
                    "rotation": first_kf["rotation"],
                    "translation": first_kf["translation"],
                    "scale": first_kf["scale"]
                }
            else:
                # 动画未定义此骨骼，保持当前状态
                next_anim_starts[bone] = deepcopy(bone_states[bone])
        
        # 如果不是第一个动画，插入过渡帧
        if idx > 0 and transition_time > 0:
            for bone in all_bones:
                prev_state = bone_states[bone]
                next_state = next_anim_starts[bone]
                
                # 生成过渡关键帧
                for step in range(1, transition_steps + 1):
                    t = step / transition_steps
                    interp_time = current_time - transition_time + (transition_time * t)
                    
                    interp_frame = create_keyframe(
                        interp_time,
                        [prev_state["rotation"][i] + (next_state["rotation"][i] - prev_state["rotation"][i]) * t for i in range(3)],
                        [prev_state["translation"][i] + (next_state["translation"][i] - prev_state["translation"][i]) * t for i in range(3)],
                        [prev_state["scale"][i] + (next_state["scale"][i] - prev_state["scale"][i]) * t for i in range(3)]
                    )
                    merged["keyframes"][bone].append(interp_frame)
        
        # 添加当前动画的关键帧
        for bone in all_bones:
            if bone in anim["keyframes"]:
                for kf in anim["keyframes"][bone]:
                    new_kf = deepcopy(kf)
                    new_kf["time"] = round(kf["time"] + current_time, 4)
                    merged["keyframes"][bone].append(new_kf)
                
                # 更新状态
                last_kf = get_last_keyframe(anim["keyframes"][bone])
                if last_kf:
                    bone_states[bone] = {
                        "rotation": deepcopy(last_kf["rotation"]),
                        "translation": deepcopy(last_kf["translation"]),
                        "scale": deepcopy(last_kf["scale"])
                    }
            else:
                # 骨骼未在此动画中定义，插入保持帧
                state = bone_states[bone]
                merged["keyframes"][bone].append(create_keyframe(
                    current_time, state["rotation"], state["translation"], state["scale"]
                ))
                merged["keyframes"][bone].append(create_keyframe(
                    current_time + anim_duration, state["rotation"], state["translation"], state["scale"]
                ))
        
        current_time += anim_duration
        
        # 添加过渡时间
        if idx < len(animations) - 1:
            current_time += transition_time
    
    merged["duration"] = round(current_time, 4)
    
    # 排序去重
    for bone in merged["keyframes"]:
        seen = {}
        unique = []
        for kf in sorted(merged["keyframes"][bone], key=lambda x: x["time"]):
            t = round(kf["time"], 4)
            if t not in seen:
                seen[t] = True
                unique.append(kf)
        merged["keyframes"][bone] = unique
    
    merged["keyframes"] = {k: v for k, v in merged["keyframes"].items() if v}
    
    return merged

def create_sequence_with_config(config_file):
    """从配置文件创建动画序列"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    animation_files = []
    for item in config["sequence"]:
        repeat = item.get("repeat", 1)
        for _ in range(repeat):
            animation_files.append(item["file"])
    
    smooth = config.get("smooth", False)
    
    if smooth:
        return merge_with_smooth_transition(
            animation_files,
            transition_time=config.get("transition_time", 0.3),
            transition_steps=config.get("transition_steps", 5),
            output_name=config.get("output_name", "merged_animation")
        )
    else:
        return merge_animations(
            animation_files,
            transition_time=config.get("transition_time", 0.3),
            output_name=config.get("output_name", "merged_animation")
        )

def main():
    parser = argparse.ArgumentParser(description="动画关键帧拼接工具（修正版）")
    parser.add_argument("files", nargs="*", help="要拼接的动画JSON文件")
    parser.add_argument("-c", "--config", help="使用配置文件指定动画序列")
    parser.add_argument("-o", "--output", default="merged_animation.json", help="输出文件名")
    parser.add_argument("-t", "--transition", type=float, default=0.3, help="过渡时间（秒）")
    parser.add_argument("-s", "--smooth", action="store_true", help="启用平滑过渡插值")
    parser.add_argument("--steps", type=int, default=5, help="平滑过渡的插值步数")
    
    args = parser.parse_args()
    
    if args.config:
        merged = create_sequence_with_config(args.config)
    elif args.files:
        if args.smooth:
            merged = merge_with_smooth_transition(
                args.files,
                transition_time=args.transition,
                transition_steps=args.steps,
                output_name=Path(args.output).stem
            )
        else:
            merged = merge_animations(
                args.files,
                transition_time=args.transition,
                output_name=Path(args.output).stem
            )
    else:
        parser.print_help()
        return
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 合并完成!")
    print(f"   输出文件: {args.output}")
    print(f"   总时长: {merged['duration']}s")
    print(f"   骨骼数量: {len(merged['keyframes'])}")

if __name__ == "__main__":
    main()
