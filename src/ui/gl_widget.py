"""
OpenGL渲染视图
"""
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QSurfaceFormat

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("⚠ 需要安装: pip install PyOpenGL PyOpenGL_accelerate")
    raise

import numpy as np
from src.utils.math_utils import Vector3


class GLWidget(QOpenGLWidget):
    """OpenGL渲染Widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 数据
        self.mesh = None
        self.skeleton = None
        self.deformer = None
        
        # 相机参数
        self.camera_distance = 3.0
        self.camera_azimuth = 90.0  # 方位角
        self.camera_elevation = 20.0  # 仰角
        self.camera_target = Vector3(0, 0, 1.0)
        
        # 鼠标交互
        self.last_mouse_pos = QPoint()
        self.is_rotating = False
        self.is_panning = False
        
        # 渲染选项
        self.show_skeleton = True
        self.show_mesh = True
        self.wireframe_mode = False
        
        # 设置OpenGL格式
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        self.setFormat(fmt)
    
    def set_data(self, mesh, skeleton, deformer):
        """设置渲染数据"""
        self.mesh = mesh
        self.skeleton = skeleton
        self.deformer = deformer
        self.update()
    
    def reset_camera(self):
        """重置相机"""
        self.camera_distance = 3.0
        self.camera_azimuth = 90.0
        self.camera_elevation = 20.0
        self.camera_target = Vector3(0, 0, 1.0)
        self.update()
    
    # ===== OpenGL回调 =====
    
    def initializeGL(self):
        """初始化OpenGL"""
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)
        
        # 光照
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # 光源设置
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
    
    def resizeGL(self, w, h):
        """窗口大小改变"""
        glViewport(0, 0, w, h)
    
    def paintGL(self):
        """绘制场景"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置投影
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width() / max(self.height(), 1)
        gluPerspective(45.0, aspect, 0.1, 100.0)
        
        # 设置视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 计算相机位置
        cam_pos = self._get_camera_position()
        gluLookAt(cam_pos.x, cam_pos.y, cam_pos.z,
                  self.camera_target.x, self.camera_target.y, self.camera_target.z,
                  0, 0, 1)
        
        # 统一旋转（让模型站起来）
        glRotatef(90, 1, 0, 0)
        
        # 绘制网格
        if self.show_mesh and self.mesh:
            self._draw_mesh()
        
        # 绘制骨架
        if self.show_skeleton and self.skeleton:
            self._draw_skeleton()
    
    def _get_camera_position(self):
        """计算相机位置（球坐标）"""
        azimuth_rad = np.radians(self.camera_azimuth)
        elevation_rad = np.radians(self.camera_elevation)
        
        x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = self.camera_distance * np.sin(elevation_rad)
        
        return self.camera_target + Vector3(x, y, z)
    
    # ===== 绘制方法 =====
    
    def _draw_mesh(self):
        """绘制网格"""
        if self.deformer:
            vertices = self.deformer.get_deformed_vertices()
        else:
            vertices = self.mesh.vertices
        
        normals = self._compute_normals(vertices)
        
        # 半透明 + 线框
        if not self.wireframe_mode:
            # 半透明面
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.8, 0.8, 0.8, 0.3)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
            glBegin(GL_TRIANGLES)
            for face in self.mesh.faces:
                for idx in face.vertex_indices:
                    n = normals[idx]
                    v = vertices[idx]
                    glNormal3f(n.x, n.y, n.z)
                    glVertex3f(v.x, v.y, v.z)
            glEnd()
            
            glDisable(GL_BLEND)
        
        # 线框
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 0.0, 0.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.0)
        
        glBegin(GL_TRIANGLES)
        for face in self.mesh.faces:
            for idx in face.vertex_indices:
                v = vertices[idx]
                glVertex3f(v.x, v.y, v.z)
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
    
    def _draw_skeleton(self):
        """绘制骨架"""
        glDisable(GL_LIGHTING)
        
        # 绘制骨骼
        glColor3f(0.0, 0.8, 1.0)
        glLineWidth(3.0)
        
        glBegin(GL_LINES)
        for bone in self.skeleton.bones:
            start = bone.start_joint.current_position
            end = bone.end_joint.current_position
            
            glVertex3f(start.x, start.y, start.z)
            glVertex3f(end.x, end.y, end.z)
        glEnd()
        
        # 绘制关节点
        glPointSize(8.0)
        glColor3f(1.0, 0.0, 0.0)
        
        glBegin(GL_POINTS)
        for joint in self.skeleton.joints:
            pos = joint.current_position
            glVertex3f(pos.x, pos.y, pos.z)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _compute_normals(self, vertices):
        """计算顶点法线"""
        num_vertices = len(vertices)
        normals = [Vector3(0, 0, 0) for _ in range(num_vertices)]
        
        for face in self.mesh.faces:
            v0 = vertices[face.vertex_indices[0]]
            v1 = vertices[face.vertex_indices[1]]
            v2 = vertices[face.vertex_indices[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = Vector3.cross(edge1, edge2)
            
            length = face_normal.length()
            if length > 1e-8:
                face_normal = face_normal * (1.0 / length)
            
            for idx in face.vertex_indices:
                normals[idx] = normals[idx] + face_normal
        
        for i in range(num_vertices):
            length = normals[i].length()
            if length > 1e-8:
                normals[i] = normals[i] * (1.0 / length)
            else:
                normals[i] = Vector3(0, 1, 0)
        
        return normals
    
    # ===== 鼠标交互 =====
    
    def mousePressEvent(self, event):
        """鼠标按下"""
        self.last_mouse_pos = event.pos()
        
        if event.button() == Qt.LeftButton:
            self.is_rotating = True
        elif event.button() == Qt.RightButton:
            self.is_panning = True
    
    def mouseReleaseEvent(self, event):
        """鼠标释放"""
        if event.button() == Qt.LeftButton:
            self.is_rotating = False
        elif event.button() == Qt.RightButton:
            self.is_panning = False
    
    def mouseMoveEvent(self, event):
        """鼠标移动"""
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        
        if self.is_rotating:
            # 旋转相机
            self.camera_azimuth -= dx * 0.5
            self.camera_elevation = np.clip(self.camera_elevation + dy * 0.5, -89, 89)
            self.update()
        
        elif self.is_panning:
            # 平移目标
            sensitivity = 0.01
            right = Vector3(np.cos(np.radians(self.camera_azimuth + 90)), 
                           np.sin(np.radians(self.camera_azimuth + 90)), 0)
            up = Vector3(0, 0, 1)
            
            self.camera_target = self.camera_target - right * (dx * sensitivity)
            self.camera_target = self.camera_target + up * (dy * sensitivity)
            self.update()
        
        self.last_mouse_pos = event.pos()
    
    def wheelEvent(self, event):
        """鼠标滚轮（缩放）"""
        delta = event.angleDelta().y()
        self.camera_distance *= 0.9 if delta > 0 else 1.1
        self.camera_distance = np.clip(self.camera_distance, 0.5, 20.0)
        self.update()

    
    def capture_frame(self):
        """
        捕获当前帧的图像
        
        Returns:
            numpy数组 (height, width, 3) RGB格式
        """
        # 确保OpenGL上下文是当前的
        self.makeCurrent()
        
        # 读取像素数据
        width = self.width()
        height = self.height()
        
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # 转换为numpy数组
        import numpy as np
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        
        # OpenGL的原点在左下角，需要上下翻转
        image = np.flipud(image)
        
        return image