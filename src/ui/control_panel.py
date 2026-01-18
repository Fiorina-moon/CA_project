"""
骨架控制面板 - 带动画播放功能
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QSlider, QPushButton, QGroupBox, 
                             QTabWidget, QListWidget, QCheckBox, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from pathlib import Path


class ControlPanel(QWidget):
    """骨架控制面板"""
    
    # 信号
    joint_transform_changed = pyqtSignal(str, tuple)  # 手动控制
    animation_selected = pyqtSignal(str)  # 动画选择
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    time_seek = pyqtSignal(float)  # 时间轴拖动
    loop_toggled = pyqtSignal(bool)
    export_video_clicked = pyqtSignal()
    render_mode_changed = pyqtSignal(str)  # 🔧 新增：渲染模式改变
    show_skeleton_toggled = pyqtSignal(bool)  # 🔧 新增：骨架显示切换

    def __init__(self, parent=None):
        super().__init__(parent)
        self.skeleton = None
        self.current_joint = None
        self.animations_dir = None
        
        # 手动控制的滑块
        self.slider_rx = None
        self.slider_ry = None
        self.slider_rz = None
        
        # 动画控制的组件
        self.animation_list = None
        self.time_slider = None
        self.time_label = None
        self.loop_checkbox = None
        
        self._init_ui()

    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("控制面板")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # 标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_manual_tab(), "手动控制")
        self.tabs.addTab(self._create_animation_tab(), "动画播放")
        layout.addWidget(self.tabs)
        
        # 视图设置
        view_group = QGroupBox("视图设置")
        view_layout = QVBoxLayout()
        
        # 渲染模式
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("渲染模式:"))
        
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems([
            "半透明+线框",
            "仅线框",
        ])
        self.render_mode_combo.currentTextChanged.connect(self._on_render_mode_changed)
        mode_layout.addWidget(self.render_mode_combo)
        
        view_layout.addLayout(mode_layout)
        
        # 显示骨架开关
        self.show_skeleton_checkbox = QCheckBox("显示骨架")
        self.show_skeleton_checkbox.setChecked(True)
        self.show_skeleton_checkbox.toggled.connect(self.show_skeleton_toggled.emit)
        view_layout.addWidget(self.show_skeleton_checkbox)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # 信息区域
        info_group = QGroupBox("信息")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("未加载骨架")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
    
    def _create_manual_tab(self):
        """创建手动控制标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 关节选择
        joint_group = QGroupBox("选择关节")
        joint_layout = QVBoxLayout()
        
        self.joint_combo = QComboBox()
        self.joint_combo.currentTextChanged.connect(self._on_joint_selected)
        joint_layout.addWidget(self.joint_combo)
        
        joint_group.setLayout(joint_layout)
        layout.addWidget(joint_group)
        
        # 旋转控制
        rotation_group = QGroupBox("旋转控制")
        rotation_layout = QVBoxLayout()
        
        # X轴
        rotation_layout.addWidget(QLabel("X轴旋转:"))
        self.slider_rx = self._create_slider()
        self.slider_rx.valueChanged.connect(self._on_rotation_changed)
        rotation_layout.addWidget(self.slider_rx)
        
        self.label_rx = QLabel("0.00°")
        self.label_rx.setAlignment(Qt.AlignCenter)
        rotation_layout.addWidget(self.label_rx)
        
        # Y轴
        rotation_layout.addWidget(QLabel("Y轴旋转:"))
        self.slider_ry = self._create_slider()
        self.slider_ry.valueChanged.connect(self._on_rotation_changed)
        rotation_layout.addWidget(self.slider_ry)
        
        self.label_ry = QLabel("0.00°")
        self.label_ry.setAlignment(Qt.AlignCenter)
        rotation_layout.addWidget(self.label_ry)
        
        # Z轴
        rotation_layout.addWidget(QLabel("Z轴旋转:"))
        self.slider_rz = self._create_slider()
        self.slider_rz.valueChanged.connect(self._on_rotation_changed)
        rotation_layout.addWidget(self.slider_rz)
        
        self.label_rz = QLabel("0.00°")
        self.label_rz.setAlignment(Qt.AlignCenter)
        rotation_layout.addWidget(self.label_rz)
        
        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)
        
        # 重置按钮
        reset_btn = QPushButton("重置姿态")
        reset_btn.clicked.connect(self._reset_pose)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        return widget
    
    def _create_animation_tab(self):
        """创建动画播放标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 动画列表
        anim_group = QGroupBox("选择动画")
        anim_layout = QVBoxLayout()
        
        self.animation_list = QListWidget()
        self.animation_list.itemClicked.connect(self._on_animation_selected)
        anim_layout.addWidget(self.animation_list)
        
        anim_group.setLayout(anim_layout)
        layout.addWidget(anim_group)
        
        # 播放控制
        control_group = QGroupBox("播放控制")
        control_layout = QVBoxLayout()
        
        # 按钮行
        button_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.play_clicked.emit)
        button_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        control_layout.addLayout(button_layout)
        
        # 时间轴
        time_layout = QVBoxLayout()
        
        self.time_label = QLabel("0.00s / 0.00s")
        self.time_label.setAlignment(Qt.AlignCenter)
        time_layout.addWidget(self.time_label)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)
        self.time_slider.setValue(0)
        self.time_slider.sliderReleased.connect(self._on_time_slider_released)
        time_layout.addWidget(self.time_slider)
        
        control_layout.addLayout(time_layout)
        
        # 循环开关
        self.loop_checkbox = QCheckBox("循环播放")
        self.loop_checkbox.setChecked(True)
        self.loop_checkbox.toggled.connect(self.loop_toggled.emit)
        control_layout.addWidget(self.loop_checkbox)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 导出视频按钮
        export_btn = QPushButton("导出视频...")
        export_btn.clicked.connect(self.export_video_clicked.emit)
        layout.addWidget(export_btn)
        
        layout.addStretch()
        return widget
    
    def _create_slider(self):
        """创建旋转滑块"""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-180)
        slider.setMaximum(180)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(30)
        return slider
    
    # ===== 手动控制相关 =====
    
    def set_skeleton(self, skeleton):
        """设置骨架"""
        self.skeleton = skeleton
        
        if skeleton:
            # 过滤主要关节
            main_joints = [j.name for j in skeleton.joints 
                          if not j.name.startswith('_') 
                          and 'Ankle' not in j.name
                          and 'Eyelid' not in j.name]
            
            self.joint_combo.clear()
            self.joint_combo.addItems(main_joints)
            
            self.info_label.setText(
                f"关节数: {skeleton.get_joint_count()}\n"
                f"骨骼数: {skeleton.get_bone_count()}"
            )
    
    def _on_joint_selected(self, joint_name):
        """关节选择"""
        if not self.skeleton or not joint_name:
            return
        
        self.current_joint = self.skeleton.joint_map.get(joint_name)
        
        # 重置滑块
        self.slider_rx.blockSignals(True)
        self.slider_ry.blockSignals(True)
        self.slider_rz.blockSignals(True)
        
        self.slider_rx.setValue(0)
        self.slider_ry.setValue(0)
        self.slider_rz.setValue(0)
        
        self.slider_rx.blockSignals(False)
        self.slider_ry.blockSignals(False)
        self.slider_rz.blockSignals(False)
        
        self._update_labels()
    
    def _on_rotation_changed(self):
        """旋转改变"""
        if not self.current_joint:
            return
        
        rx = self.slider_rx.value()
        ry = self.slider_ry.value()
        rz = self.slider_rz.value()
        
        self._update_labels()
        
        rotation = (np.radians(rx), np.radians(ry), np.radians(rz))
        self.joint_transform_changed.emit(self.current_joint.name, rotation)
    
    def _update_labels(self):
        """更新角度标签"""
        self.label_rx.setText(f"{self.slider_rx.value()}°")
        self.label_ry.setText(f"{self.slider_ry.value()}°")
        self.label_rz.setText(f"{self.slider_rz.value()}°")
    
    def _reset_pose(self):
        """重置姿态"""
        if not self.skeleton:
            return
        
        from src.utils.math_utils import Matrix4
        for joint in self.skeleton.joints:
            joint.local_transform = Matrix4.identity()
        
        self.skeleton.update_global_transforms()
        
        self.slider_rx.setValue(0)
        self.slider_ry.setValue(0)
        self.slider_rz.setValue(0)
        
        if self.current_joint:
            self.joint_transform_changed.emit(self.current_joint.name, (0, 0, 0))
    
    # ===== 动画播放相关 =====
    
    def load_animations(self, animations_dir):
        """加载动画列表"""
        self.animations_dir = Path(animations_dir)
        self.animation_list.clear()
        
        if not self.animations_dir.exists():
            return
        
        animations = sorted([f.stem for f in self.animations_dir.glob("*.json")])
        self.animation_list.addItems(animations)
        
        print(f"✓ 加载了 {len(animations)} 个动画")
    
    def _on_animation_selected(self, item):
        """动画选择"""
        anim_name = item.text()
        self.animation_selected.emit(anim_name)
    
    def _on_time_slider_released(self):
        """时间轴拖动"""
        value = self.time_slider.value()
        ratio = value / 1000.0
        self.time_seek.emit(ratio)
    
    def update_playback_time(self, current_time, total_time):
        """更新播放时间显示"""
        self.time_label.setText(f"{current_time:.2f}s / {total_time:.2f}s")
        
        # 更新时间轴（但不触发信号）
        if total_time > 0:
            ratio = current_time / total_time
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int(ratio * 1000))
            self.time_slider.blockSignals(False)
    
    def set_playing_state(self, is_playing):
        """设置播放状态"""
        self.play_btn.setEnabled(not is_playing)
        self.pause_btn.setEnabled(is_playing)
        self.stop_btn.setEnabled(is_playing or self.time_slider.value() > 0)

    def _on_render_mode_changed(self, mode_text):
        """渲染模式改变"""
        self.render_mode_changed.emit(mode_text)
    
    def get_current_render_mode(self):
        """获取当前选择的渲染模式"""
        return self.render_mode_combo.currentText()