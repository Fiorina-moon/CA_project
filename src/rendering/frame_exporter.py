"""
帧导出器
用于捕获渲染帧并导出为图片或视频
"""
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
except ImportError:
    print("⚠ OpenGL库未安装，帧捕获功能不可用")


class FrameExporter:
    """帧导出器"""
    
    def __init__(self, width: int, height: int):
        """
        初始化帧导出器
        
        Args:
            width: 帧宽度
            height: 帧高度
        """
        self.width = width
        self.height = height
    
    def capture_frame(self) -> np.ndarray:
        """
        捕获当前OpenGL帧缓冲区的内容
        
        Returns:
            RGB图像数组，形状为 (height, width, 3)
        """
        # 读取OpenGL帧缓冲区
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # 转换为numpy数组
        image = np.frombuffer(pixels, dtype=np.uint8)
        image = image.reshape(self.height, self.width, 3)
        
        # 翻转Y轴（OpenGL坐标系原点在左下角）
        image = np.flipud(image)
        
        return image
    
    def save_frame(self, image: np.ndarray, filepath: Path):
        """
        保存图像到文件
        
        Args:
            image: RGB图像数组
            filepath: 保存路径
        """
        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        img = Image.fromarray(image, 'RGB')
        img.save(filepath)
    
    @staticmethod
    def create_video(frame_dir: Path, output_path: Path, fps: int = 30):
        """
        从帧序列创建视频
        
        Args:
            frame_dir: 帧目录
            output_path: 输出视频路径
            fps: 帧率
        """
        # 优先使用OpenCV
        try:
            import cv2
            FrameExporter._create_video_opencv(frame_dir, output_path, fps)
        except ImportError:
            # 备用方案：imageio
            print("⚠ OpenCV未安装，使用imageio")
            FrameExporter._create_video_imageio(frame_dir, output_path, fps)
    
    @staticmethod
    def _create_video_opencv(frame_dir: Path, output_path: Path, fps: int):
        """使用OpenCV创建视频"""
        import cv2
        
        # 获取所有帧文件
        frames = sorted(frame_dir.glob("frame_*.png"))
        if not frames:
            print("✗ 未找到帧文件")
            return
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(str(frames[0]))
        height, width = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"\n创建视频:")
        print(f"  输入: {len(frames)} 帧")
        print(f"  输出: {output_path}")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        
        # 写入所有帧
        for i, frame_path in enumerate(frames):
            if (i + 1) % 30 == 0:
                print(f"  进度: {i + 1}/{len(frames)}")
            
            frame = cv2.imread(str(frame_path))
            out.write(frame)
        
        out.release()
        print(f"✓ 视频创建完成: {output_path}")
    
    @staticmethod
    def _create_video_imageio(frame_dir: Path, output_path: Path, fps: int):
        """使用imageio创建视频"""
        try:
            import imageio
        except ImportError:
            print("✗ 请安装: pip install opencv-python 或 pip install imageio[ffmpeg]")
            return
        
        # 获取所有帧文件
        frames = sorted(frame_dir.glob("frame_*.png"))
        if not frames:
            print("✗ 未找到帧文件")
            return
        
        print(f"\n使用imageio创建视频:")
        print(f"  帧数: {len(frames)}")
        
        # 读取所有帧
        images = [imageio.imread(str(frame_path)) for frame_path in frames]
        
        # 写入视频
        imageio.mimsave(str(output_path), images, fps=fps)
        print(f"✓ 视频创建完成: {output_path}")
