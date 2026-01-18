"""
骨架控制面板
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QSlider, QPushButton, QGroupBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np


class ControlPanel(QWidget):
    """骨架控制面板"""
    
    # 信号：关节变换改变 (joint_name, rotation_xyz)
    joint_transform_changed = pyqtSignal(str, tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.skeleton = None
        self.current_joint = None
        
        # 滑块（当前选中关节的旋转）
        self.slider_rx = None
        self.slider_ry = None
        self.slider_rz = None
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("骨架控制")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
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
        
        # X轴旋转
        rotation_layout.addWidget(QLabel("X轴旋转:"))
        self.slider_rx = self._create_slider()
        self.slider_rx.valueChanged.connect(self._on_rotation_changed)
        rotation_layout.addWidget(self.slider_rx)
        
        self.label_rx = QLabel("0.00°")
        self.label_rx.setAlignment(Qt.AlignCenter)
        rotation_layout.addWidget(self.label_rx)
        
        # Y轴旋转
        rotation_layout.addWidget(QLabel("Y轴旋转:"))
        self.slider_ry = self._create_slider()
        self.slider_ry.valueChanged.connect(self._on_rotation_changed)
        rotation_layout.addWidget(self.slider_ry)
        
        self.label_ry = QLabel("0.00°")
        self.label_ry.setAlignment(Qt.AlignCenter)
        rotation_layout.addWidget(self.label_ry)
        
        # Z轴旋转
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
        
        # 弹簧（占据剩余空间）
        layout.addStretch()
        
        # 信息显示
        info_group = QGroupBox("信息")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("未加载骨架")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
    
    def _create_slider(self):
        """创建旋转滑块 (-180° ~ +180°)"""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-180)
        slider.setMaximum(180)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(30)
        return slider
    
    def set_skeleton(self, skeleton):
        """设置骨架"""
        self.skeleton = skeleton
        
        # 填充关节列表（只显示主要关节，排除辅助关节）
        self.joint_combo.clear()
        
        if skeleton:
            # 过滤关节（排除 _rootJoint, Ankle, Eyelid 等辅助关节）
            main_joints = [j.name for j in skeleton.joints 
                          if not j.name.startswith('_') 
                          and 'Ankle' not in j.name
                          and 'Eyelid' not in j.name]
            
            self.joint_combo.addItems(main_joints)
            
            # 更新信息
            self.info_label.setText(
                f"关节数: {skeleton.get_joint_count()}\n"
                f"骨骼数: {skeleton.get_bone_count()}"
            )
    
    def _on_joint_selected(self, joint_name):
        """关节选择改变"""
        if not self.skeleton or not joint_name:
            return
        
        self.current_joint = self.skeleton.joint_map.get(joint_name)
        
        # 重置滑块（不触发信号）
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
        """旋转滑块改变"""
        if not self.current_joint:
            return
        
        # 读取滑块值（度）
        rx = self.slider_rx.value()
        ry = self.slider_ry.value()
        rz = self.slider_rz.value()
        
        # 更新标签
        self._update_labels()
        
        # 转换为弧度并发送信号
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
        
        # 重置所有关节变换
        from src.utils.math_utils import Matrix4
        for joint in self.skeleton.joints:
            joint.local_transform = Matrix4.identity()
        
        self.skeleton.update_global_transforms()
        
        # 重置滑块
        self.slider_rx.setValue(0)
        self.slider_ry.setValue(0)
        self.slider_rz.setValue(0)
        
        # 触发更新
        if self.current_joint:
            self.joint_transform_changed.emit(self.current_joint.name, (0, 0, 0))