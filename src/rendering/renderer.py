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
        self.render_mode = 'transparent'  # 'solid', 'wireframe', 'transparent', 'wireframe_transparent'
        
        # 法线缓存
        self._deformed_normals = None
    
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
        
        # 🔧 统一旋转整个场景（在这里应用一次就够了）
        glRotatef(90, 1, 0, 0)  # 让模型站起来
        
        # 渲染网格（会继承上面的旋转）
        if deformer:
            self._render_deformed_mesh(mesh, deformer)
        else:
            self._render_mesh(mesh)
        
        # 渲染骨架（也会继承旋转）
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
        """渲染变形后的网格
        
        渲染模式:
            - 'transparent_wireframe': 半透明面 + 黑色线框（默认）
            - 'solid': 灰色实体
        """
        # 获取变形后的顶点
        vertices = deformer.get_deformed_vertices()
        
        # 重新计算法线
        normals = self._compute_normals(mesh, vertices)
        
        if self.render_mode == 'solid':
            # 模式2: 灰色实体
            glColor3f(0.7, 0.7, 0.7)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for i, idx in enumerate(face.vertex_indices):
                    v = vertices[idx]
                    n = normals[idx]
                    glNormal3f(n.x, n.y, n.z)
                    glVertex3f(v.x, v.y, v.z)
            glEnd()
            
        else:  # 'transparent_wireframe' 或默认
            # 模式1: 半透明面 + 黑色线框
            
            # 先画半透明面
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor3f(0.8, 0.8, 0.8)  # 浅灰色，30%透明度
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for i, idx in enumerate(face.vertex_indices):
                    v = vertices[idx]
                    n = normals[idx]
                    glNormal3f(n.x, n.y, n.z)
                    glVertex3f(v.x, v.y, v.z)
            glEnd()
            
            glDisable(GL_BLEND)
            
            # 再画黑色线框
            glDisable(GL_LIGHTING)
            glColor3f(0.0, 0.0, 0.0)  # 黑色
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
            
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for i, idx in enumerate(face.vertex_indices):
                    v = vertices[idx]
                    glVertex3f(v.x, v.y, v.z)
            glEnd()
            
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)

    
    def _compute_normals(self, mesh: Mesh, vertices: List[Vector3]) -> List[Vector3]:
        """
        为变形后的顶点重新计算法线
        
        方法：对每个顶点，平均其相邻面的法线
        """
        num_vertices = len(vertices)
        normals = [Vector3(0, 0, 0) for _ in range(num_vertices)]
        
        # 对每个面计算法线
        for face in mesh.faces:
            v0 = vertices[face.vertex_indices[0]]
            v1 = vertices[face.vertex_indices[1]]
            v2 = vertices[face.vertex_indices[2]]
            
            # 计算面法线（叉积）
            edge1 = v1 - v0
            edge2 = v2 - v0
            
            face_normal = Vector3.cross(edge1, edge2)
            length = face_normal.length()
            
            if length > 1e-8:
                face_normal = face_normal * (1.0 / length)  # 归一化
            
            # 累加到顶点法线
            for idx in face.vertex_indices:
                normals[idx] = normals[idx] + face_normal
        
        # 归一化顶点法线
        for i in range(num_vertices):
            length = normals[i].length()
            if length > 1e-8:
                normals[i] = normals[i] * (1.0 / length)
            else:
                normals[i] = Vector3(0, 1, 0)  # 默认法线
        
        return normals
    
    def _render_skeleton(self, skeleton: Skeleton):
        """渲染骨架 - 与模型保持一致的坐标系"""
        glDisable(GL_LIGHTING)
        
        # 🔧 保存当前矩阵状态
        glPushMatrix()
        
        # 🔧 应用与模型相同的旋转（如果你在 render_frame 里有 glRotatef）
        # 注意：这里不需要再次旋转，因为已经在 render_frame 里统一旋转了
        # 如果你之前在 render_frame 的第96行有 glRotatef(-90, 1, 0, 0)
        # 那么骨架会自动跟着旋转
        
        # 绘制骨骼
        glColor3f(0.0, 0.8, 1.0)
        glLineWidth(3.0)
        
        glBegin(GL_LINES)
        for bone in skeleton.bones:
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
        
        glPopMatrix()  # 🔧 恢复矩阵状态
        
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