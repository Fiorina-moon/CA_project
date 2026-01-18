"""
视频导出对话框 - 直接录制UI画面
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSpinBox, QGroupBox, 
                             QFileDialog, QMessageBox, QProgressDialog, 
                             QApplication, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer
from pathlib import Path
import numpy as np
import cv2
from src.config import VIDEOS_DIR


class VideoExportDialog(QDialog):
    """视频导出对话框 - 录制UI画面"""
    
    def __init__(self, parent, animation_name, animator, deformer, gl_widget):
        super().__init__(parent)
        self.animation_name = animation_name
        self.animator = animator
        self.deformer = deformer
        self.gl_widget = gl_widget
        
        # 录制状态
        self.is_recording = False
        self.frames = []
        self.current_time = 0.0
        self.target_duration = 0.0
        self.animation_duration = 0.0
        self.timer = QTimer()
        self.timer.timeout.connect(self._capture_frame)
        
        # 保存原始状态
        self.original_loop = None
        
        self.setWindowTitle("导出视频")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 动画信息
        info = QLabel(f"动画: {self.animation_name}")
        info.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(info)
        
        if self.animator.current_clip:
            duration_label = QLabel(f"动画时长: {self.animator.current_clip.duration:.2f}秒")
            layout.addWidget(duration_label)
        
        # 时长设置
        duration_group = QGroupBox("导出时长")
        duration_layout = QVBoxLayout()
        
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("时长:"))
        
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 60.0)
        self.duration_spin.setValue(2.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setSuffix(" 秒")
        time_layout.addWidget(self.duration_spin)
        
        time_layout.addStretch()
        duration_layout.addLayout(time_layout)
        
        duration_group.setLayout(duration_layout)
        layout.addWidget(duration_group)
        
        # 视频设置
        settings_group = QGroupBox("视频设置")
        settings_layout = QVBoxLayout()
        
        # 帧率
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("帧率:"))
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(24, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" FPS")
        fps_layout.addWidget(self.fps_spin)
        
        fps_layout.addStretch()
        settings_layout.addLayout(fps_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 说明
        note = QLabel("提示: 将录制左侧OpenGL视图的画面，如果时长超过动画时长，动画将循环播放")
        note.setStyleSheet("color: #666; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)
        
        # 输出路径
        output_group = QGroupBox("输出路径")
        output_layout = QVBoxLayout()
        
        path_layout = QHBoxLayout()
        default_name = self.animation_name.replace(' ', '_')
        self.path_label = QLabel(str(VIDEOS_DIR / f"{default_name}.mp4"))
        self.path_label.setWordWrap(True)
        path_layout.addWidget(self.path_label)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_output)
        path_layout.addWidget(browse_btn)
        
        output_layout.addLayout(path_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("开始录制")
        self.export_btn.clicked.connect(self._start_recording)
        button_layout.addWidget(self.export_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _browse_output(self):
        """浏览输出路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存视频", 
            self.path_label.text(),
            "Video Files (*.mp4)"
        )
        
        if file_path:
            self.path_label.setText(file_path)
    
    def _start_recording(self):
        """开始录制"""
        if not self.animator.current_clip:
            QMessageBox.warning(self, "警告", "没有加载动画")
            return
        
        # 直接从输入框获取时长
        self.target_duration = self.duration_spin.value()
        self.animation_duration = self.animator.current_clip.duration
        
        # 准备录制
        self.frames = []
        self.current_time = 0.0
        self.is_recording = True
        self.export_btn.setEnabled(False)
        
        # 保存原始循环设置，但录制时不使用animator的循环功能
        self.original_loop = self.animator.loop
        self.animator.loop = False  # 禁用自动循环，手动控制
        
        # 重置动画到起点
        self.animator.stop()
        self.animator.set_time(0)
        if self.deformer:
            self.deformer.update()
        self.gl_widget.update()
        QApplication.processEvents()
        
        # 计算录制参数
        fps = self.fps_spin.value()
        self.expected_frames = int(self.target_duration * fps)
        
        # 开始录制
        interval_ms = int(1000.0 / fps)
        self.timer.start(interval_ms)
        
        print(f"\n开始录制: {self.animation_name}")
        print(f"目标时长: {self.target_duration:.2f}秒")
        print(f"动画时长: {self.animation_duration:.2f}秒")
        print(f"帧率: {fps} FPS")
        print(f"预计帧数: {self.expected_frames}")
        if self.target_duration > self.animation_duration:
            cycles = self.target_duration / self.animation_duration
            print(f"将循环播放: {cycles:.2f} 次")
    
    def _capture_frame(self):
        """捕获一帧"""
        if not self.is_recording:
            return
        
        # 计算当前应该在的时间点（精确控制）
        fps = self.fps_spin.value()
        dt = 1.0 / fps
        self.current_time += dt
        
        # 检查是否录制完成
        if self.current_time >= self.target_duration:
            print(f"录制完成: 时间 {self.current_time:.2f}/{self.target_duration:.2f}秒")
            self._finish_recording()
            return
        
        # 手动循环：计算当前在动画中的位置
        animation_time = self.current_time % self.animation_duration
        
        # 检测是否需要重新开始循环
        if animation_time < dt:  # 刚刚回到开头
            cycle_num = int(self.current_time / self.animation_duration) + 1
            print(f"  循环: 第 {cycle_num} 轮")
        
        # 设置动画到指定时间点
        self.animator.set_time(animation_time)
        
        if self.deformer:
            self.deformer.update()
        
        self.gl_widget.update()
        QApplication.processEvents()
        
        # 捕获画面
        frame = self.gl_widget.capture_frame()
        self.frames.append(frame)
        
        current_frame = len(self.frames)
        
        # 打印进度（每30帧）
        if current_frame % 30 == 0 or current_frame == self.expected_frames:
            progress = (current_frame / self.expected_frames) * 100
            cycle = int(self.current_time / self.animation_duration) + 1
            print(f"录制进度: {current_frame}/{self.expected_frames} 帧 ({progress:.1f}%) - 时间: {self.current_time:.2f}s (第{cycle}轮)")
    
    def _finish_recording(self):
        """完成录制，合成视频"""
        self.timer.stop()
        self.is_recording = False
        
        # 恢复原始循环设置
        if self.original_loop is not None:
            self.animator.loop = self.original_loop
        
        print(f"\n录制完成，共 {len(self.frames)} 帧")
        
        if len(self.frames) == 0:
            QMessageBox.warning(self, "警告", "没有录制到任何帧")
            self.export_btn.setEnabled(True)
            return
        
        # 显示进度对话框
        progress = QProgressDialog("正在合成视频...", None, 0, len(self.frames), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("合成中")
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        try:
            # 合成视频
            output_path = Path(self.path_label.text())
            fps = self.fps_spin.value()
            
            self._create_video(output_path, fps, progress)
            
            progress.close()
            
            actual_duration = len(self.frames) / fps
            
            QMessageBox.information(self, "成功", 
                f"视频已导出到:\n{output_path}\n\n"
                f"总帧数: {len(self.frames)}\n"
                f"实际时长: {actual_duration:.2f}秒\n"
                f"帧率: {fps} FPS")
            self.accept()
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "错误", f"视频合成失败:\n{e}")
            import traceback
            traceback.print_exc()
            self.export_btn.setEnabled(True)
    
    def _create_video(self, output_path, fps, progress=None):
        """
        合成视频
        
        Args:
            output_path: 输出路径
            fps: 帧率
            progress: 进度对话框（可选）
        """
        if not self.frames:
            raise ValueError("没有帧数据")
        
        # 获取尺寸
        height, width, _ = self.frames[0].shape
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("无法创建视频文件")
        
        print(f"\n合成视频:")
        print(f"  分辨率: {width}×{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {len(self.frames)}")
        
        # 写入帧
        for i, frame in enumerate(self.frames):
            # RGB转BGR (OpenCV使用BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
            if progress:
                progress.setValue(i + 1)
                if i % 30 == 0:
                    QApplication.processEvents()
        
        out.release()
        print(f"✓ 视频已保存: {output_path}")