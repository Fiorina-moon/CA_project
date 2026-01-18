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
        """导出骨架结构"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存骨架", "skeleton.json", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        # 构建数据
        data = {
            "joints": [],
            "hierarchy": {}
        }
        
        for joint in self.skeleton.joints:
            data["joints"].append({
                "name": joint.name,
                "index": joint.index,
                "head": [joint.head.x, joint.head.y, joint.head.z],
                "tail": [joint.tail.x, joint.tail.y, joint.tail.z],
                "parent": joint.parent_name
            })
            
            if joint.parent_name:
                data["hierarchy"][joint.name] = joint.parent_name
        
        # 保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _export_weights(self):
        """导出蒙皮权重"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存权重", "weights.npz", "NPZ Files (*.npz)"
        )
        
        if not file_path:
            return
        
        # 保存
        np.savez_compressed(file_path, weights=self.weights)
    
    def _export_pose(self):
        """导出当前姿态"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存姿态", "pose.json", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        # 构建数据（导出每个关节的局部变换矩阵）
        data = {"joints": {}}
        
        for joint in self.skeleton.joints:
            # 将4x4矩阵转为列表
            matrix = joint.local_transform.data.tolist()
            data["joints"][joint.name] = {
                "local_transform": matrix,
                "position": [
                    joint.current_position.x,
                    joint.current_position.y,
                    joint.current_position.z
                ]
            }
        
        # 保存
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)