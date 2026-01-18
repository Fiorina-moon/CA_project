"""
主窗口
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QMenuBar, QAction, QFileDialog, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
import numpy as np

from src.ui.gl_widget import GLWidget
from src.ui.control_panel import ControlPanel
from src.ui.export_dialog import ExportDialog
from src.utils.file_io import load_animation
from src.rendering.video_export import VideoExporter
from src.ui.video_export_dialog import VideoExportDialog
from src.core.mesh_loader import OBJLoader
from src.core.skeleton_loader import SkeletonLoader 
from src.animation.animator import Animator
from src.skinning.deformer import SkinDeformer
from src.config import *


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("骨架绑定系统 - 交互模式")
        self.setGeometry(100, 100, 1400, 800)
        
        # 数据
        self.mesh = None
        self.skeleton = None
        self.deformer = None
        self.animator = None
        self.weights = None
        
        # UI组件
        self.gl_widget = None
        self.control_panel = None
        self.video_exporter = None
        
        # 定时器（动画播放）
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        
        self._init_ui()
        self._load_default_data()
    
    def _init_ui(self):
        """初始化UI"""
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：OpenGL视图（先创建）
        self.gl_widget = GLWidget(self)
        splitter.addWidget(self.gl_widget)
        
        # 右侧：控制面板
        self.control_panel = ControlPanel(self)
        self.control_panel.joint_transform_changed.connect(self._on_joint_transform_changed)
        self.control_panel.animation_selected.connect(self._on_animation_selected)
        self.control_panel.play_clicked.connect(self._on_play)
        self.control_panel.pause_clicked.connect(self._on_pause)
        self.control_panel.stop_clicked.connect(self._on_stop)
        self.control_panel.time_seek.connect(self._on_time_seek)
        self.control_panel.loop_toggled.connect(self._on_loop_toggled)
        self.control_panel.export_video_clicked.connect(self._on_export_video)
        self.control_panel.render_mode_changed.connect(self._on_render_mode_changed)
        self.control_panel.show_skeleton_toggled.connect(self._on_show_skeleton_toggled)
        splitter.addWidget(self.control_panel)
        
        # 设置分割比例
        splitter.setSizes([1000, 400])
        
        main_layout.addWidget(splitter)
        
        # 创建菜单栏（放在最后，因为需要 gl_widget 已存在）
        self._create_menu_bar()
        
        self.statusBar().showMessage("就绪")
    
    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        load_action = QAction('加载模型...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self._load_model)
        file_menu.addAction(load_action)
        
        load_weights_action = QAction('加载权重...', self)
        load_weights_action.triggered.connect(self._load_weights)
        file_menu.addAction(load_weights_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('导出数据...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self._export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        
        reset_camera_action = QAction('重置相机', self)
        reset_camera_action.setShortcut('R')
        reset_camera_action.triggered.connect(self.gl_widget.reset_camera)
        view_menu.addAction(reset_camera_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        
        about_action = QAction('关于', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _load_default_data(self):
        """加载默认数据"""
        try:
            # 加载模型
            self.mesh = OBJLoader.load(ELK_OBJ_PATH)
            
            # 加载骨架
            self.skeleton = SkeletonLoader.load(SKELETON_JSON_PATH)
            
            # 加载权重
            weights_path = WEIGHTS_DIR / "elk_weights.npz"
            if weights_path.exists():
                data = np.load(str(weights_path))
                self.weights = data['weights']
                
                # 创建变形器
                self.deformer = SkinDeformer(self.mesh, self.skeleton, self.weights)
                self.deformer.update()
            
            # 创建动画控制器
            self.animator = Animator(self.skeleton)
            
            # 更新UI
            self.gl_widget.set_data(self.mesh, self.skeleton, self.deformer)
            self.control_panel.set_skeleton(self.skeleton)

            self.control_panel.load_animations(ANIMATIONS_DIR)
            self.video_exporter = VideoExporter(
                ELK_OBJ_PATH,
                SKELETON_JSON_PATH,
                WEIGHTS_DIR / "elk_weights.npz",
                ANIMATIONS_DIR
            )
            
            self.statusBar().showMessage(f"✓ 已加载: {self.mesh.get_vertex_count()}顶点, {self.skeleton.get_joint_count()}关节")
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法加载默认数据:\n{e}")
    
    def _load_model(self):
        """加载OBJ模型"""
        file_path, _ = QFileDialog.getOpenFileName(self, "加载模型", str(MODELS_DIR), "OBJ Files (*.obj)")
        if file_path:
            try:
                self.mesh = OBJLoader.load(Path(file_path))  
                self.gl_widget.set_data(self.mesh, self.skeleton, self.deformer)
                self.statusBar().showMessage(f"✓ 已加载模型: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败:\n{e}")
    
    def _load_weights(self):
        """加载权重文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "加载权重", str(WEIGHTS_DIR), "NPZ Files (*.npz)")
        if file_path and self.mesh and self.skeleton:
            try:
                data = np.load(file_path)
                self.weights = data['weights']
                self.deformer = SkinDeformer(self.mesh, self.skeleton, self.weights)
                self.deformer.update()
                self.gl_widget.set_data(self.mesh, self.skeleton, self.deformer)
                self.statusBar().showMessage(f"✓ 已加载权重: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败:\n{e}")
    
    def _export_data(self):
        """打开导出对话框"""
        if not self.skeleton:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return
        
        dialog = ExportDialog(self.skeleton, self.weights, self)
        dialog.exec_()
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
                         "骨架绑定系统 v1.0\n\n"
                         "计算机动画大作业\n"
                         "支持手动骨架控制和数据导出")
    
    def _on_joint_transform_changed(self, joint_name, rotation):
        """关节变换改变"""
        if not self.skeleton or not self.deformer:
            return
        
        joint = self.skeleton.joint_map.get(joint_name)
        if joint:
            # 🔧 修复轴顺序：滑块(X,Y,Z) → 欧拉角(X,Y,Z)
            # 如果你发现Y和Z反了，可能需要交换
            rx, ry, rz = rotation
            
            # 方案1：直接使用（如果XYZ都对）
            from src.utils.math_utils import Matrix4
            joint.local_transform = Matrix4.from_euler(rx, rz, ry)
            
            # 更新全局变换
            self.skeleton.update_global_transforms()
            
            # 更新蒙皮变形
            self.deformer.update()
            
            # 刷新渲染
            self.gl_widget.update()
    
    def _on_timer(self):
        """定时器回调"""
        if self.animator and self.animator.is_playing:
            self.animator.update(1.0 / 30.0)
            if self.deformer:
                self.deformer.update()
            self.gl_widget.update()
            
            # 更新时间显示
            if self.animator.current_clip:
                self.control_panel.update_playback_time(
                    self.animator.current_time,
                    self.animator.current_clip.duration
                )


    def _on_animation_selected(self, anim_name):
        """动画选择"""
        try:
            anim_path = ANIMATIONS_DIR / f"{anim_name}.json"
            animation = load_animation(anim_path)
            
            self.animator.load_clip(animation)
            self.statusBar().showMessage(f"✓ 已加载动画: {anim_name} ({animation.duration:.2f}s)")
            
            # 更新时间显示
            self.control_panel.update_playback_time(0, animation.duration)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载动画失败:\n{e}")

    def _on_play(self):
        """播放"""
        if self.animator and self.animator.current_clip:
            self.animator.play()
            self.timer.start(33)  # 30 FPS
            self.control_panel.set_playing_state(True)
            self.statusBar().showMessage("播放中...")

    def _on_pause(self):
        """暂停"""
        if self.animator:
            self.animator.pause()
            self.timer.stop()
            self.control_panel.set_playing_state(False)
            self.statusBar().showMessage("已暂停")

    def _on_stop(self):
        """停止"""
        if self.animator:
            self.animator.stop()
            self.timer.stop()
            
            # 重置骨架姿态
            from src.utils.math_utils import Matrix4
            for joint in self.skeleton.joints:
                joint.local_transform = Matrix4.identity()
            self.skeleton.update_global_transforms()
            
            if self.deformer:
                self.deformer.update()
            
            self.gl_widget.update()
            self.control_panel.set_playing_state(False)
            
            if self.animator.current_clip:
                self.control_panel.update_playback_time(0, self.animator.current_clip.duration)
            
            self.statusBar().showMessage("已停止")

    def _on_time_seek(self, ratio):
        """时间轴拖动"""
        if self.animator and self.animator.current_clip:
            target_time = self.animator.current_clip.duration * ratio
            self.animator.set_time(target_time)
            
            if self.deformer:
                self.deformer.update()
            
            self.gl_widget.update()

    def _on_loop_toggled(self, checked):
        """循环开关"""
        if self.animator:
            self.animator.loop = checked

    def _on_export_video(self):
        """导出视频 - 使用新的录制方式"""
        if not self.animator or not self.animator.current_clip:
            QMessageBox.warning(self, "警告", "请先选择一个动画")
            return
        
        # 使用新的导出对话框
        from src.ui.video_export_dialog import VideoExportDialog
        
        dialog = VideoExportDialog(
            self,
            self.animator.current_clip.name,
            self.animator,
            self.deformer,
            self.gl_widget
        )
        dialog.exec_()

    def _on_render_mode_changed(self, mode_text):
        """渲染模式改变"""
        # 根据选择的模式更新 GL 视图
        if mode_text == "半透明+线框":
            self.gl_widget.show_mesh = True
            self.gl_widget.wireframe_mode = False 
        elif mode_text == "仅线框":
            self.gl_widget.show_mesh = True
            self.gl_widget.wireframe_mode = True
        
        self.gl_widget.update()
        self.statusBar().showMessage(f"渲染模式: {mode_text}")

    def _on_show_skeleton_toggled(self, checked):
        """骨架显示切换"""
        self.gl_widget.show_skeleton = checked
        self.gl_widget.update()
        self.statusBar().showMessage(f"骨架显示: {'开' if checked else '关'}")