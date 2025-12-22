"""
OpenGL渲染器
"""
import numpy as np
from typing import List
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import glfw
except ImportError:
    print("⚠ OpenGL库未安装，请运行: pip install PyOpenGL PyOpenGL_accelerate glfw")
    raise

from core.mesh import Mesh
from core.skeleton import Skeleton
from skinning.deformer import SkinDeformer
from rendering.camera import Camera
from utils.math_utils import Vector3


class Renderer:
    """OpenGL渲染器"""
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "Skeletal Animation"):
        """
        Args:
            width: 窗口宽度
            height: 窗口高度
            title: 窗口标题
        """
        self.width = width
        self.height = height
        self.title = title
        
        self.window = None
        self.camera = Camera(distance=3.0, azimuth=45, elevation=30)
        
        # 渲染选项
        self.show_wireframe = False
        self.show_skeleton = True
        self.background_color = (0.2, 0.2, 0.2, 1.0)
    
    def initialize(self) -> bool:
        """初始化OpenGL"""
        if not glfw.init():
            print("✗ GLFW初始化失败")
            return False
        
        # 创建窗口
        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            print("✗ 窗口创建失败")
            return False
        
        glfw.make_context_current(self.window)
        
        # OpenGL设置
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # 光照
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        print(f"✓ OpenGL渲染器初始化成功")
        print(f"  版本: {glGetString(GL_VERSION).decode()}")
        
        return True
    
    def render_frame(self, mesh: Mesh, deformer: SkinDeformer = None, skeleton: Skeleton = None):
        """渲染一帧"""
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置投影
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width / self.height
        gluPerspective(self.camera.fov, aspect, self.camera.near, self.camera.far)
        
        # 设置视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        pos = self.camera.get_position()
        target = self.camera.target
        gluLookAt(pos.x, pos.y, pos.z,
                  target.x, target.y, target.z,
                  0, 0, 1)
        
        # 渲染网格
        if deformer:
            self._render_deformed_mesh(mesh, deformer)
        else:
            self._render_mesh(mesh)
        
        # 渲染骨架
        if self.show_skeleton and skeleton:
            self._render_skeleton(skeleton)
        
        glfw.swap_buffers(self.window)
    
    def _render_mesh(self, mesh: Mesh):
        """渲染网格（原始）"""
        glColor3f(0.8, 0.8, 0.8)
        
        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face.vertex_indices:
                v = mesh.vertices[idx]
                
                # 法线
                if face.normal_indices and idx < len(mesh.normals):
                    n = mesh.normals[face.normal_indices[face.vertex_indices.index(idx)]]
                    glNormal3f(n.x, n.y, n.z)
                
                glVertex3f(v.x, v.y, v.z)
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def _render_deformed_mesh(self, mesh: Mesh, deformer: SkinDeformer):
        """渲染变形后的网格"""
        vertices = deformer.get_deformed_vertices()
        
        glColor3f(0.8, 0.6, 0.4)
        
        if self.show_wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face.vertex_indices:
                v = vertices[idx]
                
                # 简单法线（可优化）
                if mesh.normals and idx < len(mesh.normals):
                    n = mesh.normals[idx]
                    glNormal3f(n.x, n.y, n.z)
                
                glVertex3f(v.x, v.y, v.z)
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def _render_skeleton(self, skeleton: Skeleton):
        """渲染骨架 - 使用简化的当前位置"""
        glDisable(GL_LIGHTING)
        
        # 绘制骨骼
        glColor3f(0.0, 0.8, 1.0)
        glLineWidth(3.0)
        
        glBegin(GL_LINES)
        for bone in skeleton.bones:
            # 使用当前位置
            start = bone.start_joint.current_position
            end = bone.end_joint.current_position
            
            glVertex3f(start.x, start.y, start.z)
            glVertex3f(end.x, end.y, end.z)
        glEnd()
        
        # 绘制关节点
        glPointSize(8.0)
        glColor3f(1.0, 0.0, 0.0)
        
        glBegin(GL_POINTS)
        for joint in skeleton.joints:
            pos = joint.current_position
            glVertex3f(pos.x, pos.y, pos.z)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def should_close(self) -> bool:
        """检查窗口是否应该关闭"""
        return glfw.window_should_close(self.window)
    
    def poll_events(self):
        """处理事件"""
        glfw.poll_events()
    
    def cleanup(self):
        """清理资源"""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()