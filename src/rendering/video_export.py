"""
视频导出模块
"""
from pathlib import Path
from src.config import FRAMES_DIR, VIDEOS_DIR
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader
from src.skinning.deformer import SkinDeformer
from src.animation.animator import Animator
from src.rendering.renderer import Renderer
from src.rendering.frame_exporter import FrameExporter
from src.utils.file_io import load_weights_npz, load_animation
from src.utils.math_utils import Vector3


class VideoExporter:
    """视频导出器"""
    
    def __init__(self, mesh_path, skeleton_path, weights_path, animations_dir):
        """
        初始化导出器
        
        Args:
            mesh_path: 模型文件路径
            skeleton_path: 骨架文件路径
            weights_path: 权重文件路径
            animations_dir: 动画文件夹路径
        """
        self.mesh_path = mesh_path
        self.skeleton_path = skeleton_path
        self.weights_path = weights_path
        self.animations_dir = animations_dir
        
    def export(self, animation_name, output_path=None, 
               view_angle=90, render_mode='transparent_with_wireframe', 
               fps=30, duration=0, width=800, height=600):
        """
        导出动画视频
        
        Args:
            animation_name: 动画名称
            output_path: 输出路径（可选）
            view_angle: 相机方位角
            render_mode: 渲染模式
            fps: 帧率
            duration: 时长（0=完整）
            width: 视频宽度
            height: 视频高度
        
        Returns:
            bool: 是否成功
        """
        print("\n" + "=" * 60)
        print(f"导出动画视频: {animation_name}")
        print(f"渲染模式: {render_mode}")
        print(f"视角: {view_angle} 度")
        print(f"帧率: {fps} FPS")
        print("=" * 60)
        
        try:
            # 加载资源
            print("\n加载资源...")
            mesh = OBJLoader.load(self.mesh_path)
            skeleton = SkeletonLoader.load(self.skeleton_path)
            weights = load_weights_npz(self.weights_path)
            
            # 加载动画
            if not animation_name.endswith('.json'):
                animation_name += '.json'
            
            anim_path = self.animations_dir / animation_name
            if not anim_path.exists():
                print(f"\n错误: 文件不存在: {anim_path}")
                return False
            
            animation = load_animation(anim_path)
            print(f"\n动画信息:")
            print(f"  名称: {animation.name}")
            print(f"  时长: {animation.duration} 秒")
            
            # 初始化动画系统
            print("\n初始化动画系统...")
            deformer = SkinDeformer(mesh, skeleton, weights)
            animator = Animator(skeleton)
            animator.load_clip(animation)
            animator.play()
            
            # 初始化渲染器
            renderer = Renderer(width, height, f"Exporting - {animation.name}")
            if not renderer.initialize():
                return False
            
            renderer.render_mode = render_mode
            
            # 设置相机
            self._setup_camera(renderer, mesh, view_angle)
            
            # 渲染帧
            total_frames = self._render_frames(
                renderer, mesh, deformer, animator, 
                animation, fps, duration, width, height
            )
            
            renderer.cleanup()
            
            # 合成视频
            if output_path is None:
                output_name = animation.name.replace(' ', '_')
                output_path = VIDEOS_DIR / f"{output_name}.mp4"
            
            print(f"\n合成视频...")
            FrameExporter.create_video(FRAMES_DIR, output_path, fps)
            
            print("\n" + "=" * 60)
            print("视频导出成功!")
            print(f"文件位置: {output_path}")
            print(f"时长: {duration if duration > 0 else animation.duration} 秒")
            print(f"帧率: {fps} FPS")
            print(f"总帧数: {total_frames}")
            print("=" * 60 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n错误: 导出失败 - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_camera(self, renderer, mesh, view_angle):
        """设置相机参数"""
        min_pos, max_pos = mesh.get_bounding_box()
        center = Vector3(
            (min_pos.x + max_pos.x) / 2,
            (min_pos.y + max_pos.y) / 2,
            (min_pos.z + max_pos.z) / 2
        )
        size = max(max_pos.x - min_pos.x, max_pos.y - min_pos.y, max_pos.z - min_pos.z)
        
        adjusted_center = Vector3(center.x, center.y - 1.2, center.z + 1.2)
        renderer.camera.target = adjusted_center
        renderer.camera.distance = size * 2.0
        renderer.camera.azimuth = view_angle
        renderer.camera.elevation = 0
        
        print(f"\n相机设置:")
        print(f"  方位角: {view_angle} 度")
        print(f"  距离: {renderer.camera.distance:.2f}")
    
    def _render_frames(self, renderer, mesh, deformer, animator, 
                       animation, fps, duration, width, height):
        """渲染所有帧"""
        if duration <= 0:
            duration = animation.duration
        
        total_frames = int(duration * fps)
        dt = 1.0 / fps
        
        print(f"\n导出设置:")
        print(f"  总时长: {duration} 秒")
        print(f"  总帧数: {total_frames}")
        
        # 清空旧帧
        print(f"\n清理旧帧...")
        for old_frame in FRAMES_DIR.glob("frame_*.png"):
            old_frame.unlink()
        
        print(f"\n开始渲染帧...")
        
        exporter = FrameExporter(width, height)
        
        for frame_idx in range(total_frames):
            if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
                progress = (frame_idx + 1) / total_frames * 100
                print(f"  进度: {frame_idx + 1}/{total_frames} ({progress:.1f}%)")
            
            animator.update(dt)
            deformer.update()
            renderer.render_frame(mesh, deformer, animator.skeleton)
            
            image = exporter.capture_frame()
            frame_path = FRAMES_DIR / f"frame_{frame_idx:04d}.png"
            exporter.save_frame(image, frame_path)
            
            renderer.poll_events()
        
        print(f"\n帧渲染完成: {total_frames} 帧")
        return total_frames
