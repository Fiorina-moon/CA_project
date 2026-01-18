"""
数据导出对话框
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFileDialog, QMessageBox, QGroupBox, QRadioButton)
from PyQt5.QtCore import Qt
import json
import numpy as np
from pathlib import Path


class ExportDialog(QDialog):
    """导出对话框"""
    
    def __init__(self, skeleton, weights, parent=None):
        super().__init__(parent)
        self.skeleton = skeleton
        self.weights = weights
        
        self.setWindowTitle("导出数据")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 说明
        info = QLabel("选择要导出的数据类型：")
        layout.addWidget(info)
        
        # 导出选项
        options_group = QGroupBox("导出选项")
        options_layout = QVBoxLayout()
        
        self.radio_skeleton = QRadioButton("骨架结构 (JSON)")
        self.radio_skeleton.setChecked(True)
        options_layout.addWidget(self.radio_skeleton)
        
        self.radio_weights = QRadioButton("蒙皮权重 (NPZ)")
        self.radio_weights.setEnabled(self.weights is not None)
        options_layout.addWidget(self.radio_weights)
        
        self.radio_pose = QRadioButton("当前姿态 (JSON)")
        options_layout.addWidget(self.radio_pose)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("导出")
        export_btn.clicked.connect(self._export)
        button_layout.addWidget(export_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _export(self):
        """执行导出"""
        try:
            if self.radio_skeleton.isChecked():
                self._export_skeleton()
            elif self.radio_weights.isChecked():
                self._export_weights()
            elif self.radio_pose.isChecked():
                self._export_pose()
            
            QMessageBox.information(self, "成功", "数据已导出")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")
        
    def _export_skeleton(self):
        """导出骨架结构（应用显示旋转）"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存骨架", "skeleton.json", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        import numpy as np
        
        # 🔧 应用渲染旋转（绕X轴旋转90度）
        angle = np.radians(90)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
        
        def apply_rotation(pos):
            """应用旋转变换到位置"""
            vec = np.array([pos.x, pos.y, pos.z])
            rotated = rotation_matrix @ vec
            return [float(rotated[0]), float(rotated[1]), float(rotated[2])]
        
        # 构建数据
        data = {
            "joints": [],
            "hierarchy": {}
        }
        
        for joint in self.skeleton.joints:
            data["joints"].append({
                "name": joint.name,
                "index": int(joint.index),
                "head": apply_rotation(joint.head),  # 🔧 应用旋转
                "tail": apply_rotation(joint.tail),  # 🔧 应用旋转
                "parent": joint.parent_name
            })
            
            if joint.parent_name:
                data["hierarchy"][joint.name] = joint.parent_name
        
        # 保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 骨架已导出（已应用显示旋转）: {file_path}")


    def _export_pose(self):
        """导出当前姿态（应用显示旋转）"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存姿态", "pose.json", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        import numpy as np
        
        # 🔧 应用渲染旋转（绕X轴旋转90度）
        angle = np.radians(90)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
        
        def apply_rotation(pos):
            """应用旋转变换到位置"""
            vec = np.array([pos.x, pos.y, pos.z])
            rotated = rotation_matrix @ vec
            return [float(rotated[0]), float(rotated[1]), float(rotated[2])]
        
        # 构建数据
        data = {"joints": {}}
        
        for joint in self.skeleton.joints:
            # 局部变换矩阵（不需要旋转，这是相对变换）
            matrix = [[float(x) for x in row] for row in joint.local_transform.data.tolist()]
            
            data["joints"][joint.name] = {
                "local_transform": matrix,
                "position": apply_rotation(joint.current_position)  # 🔧 应用旋转
            }
        
        # 保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ 姿态已导出（已应用显示旋转）: {file_path}")