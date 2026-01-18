"""
OpenGL渲染器
基于 PyOpenGL 和 GLFW 实现骨骼动画可视化
"""
from typing import List, Optional

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import glfw
except ImportError:
    print("⚠ OpenGL库未安装，请运行: pip install PyOpenGL PyOpenGL_accelerate glfw")
    raise

from src.core.mesh import Mesh
from src.core.skeleton import Skeleton
from src.skinning.deformer import SkinDeformer
from src.rendering.camera import Camera
from src.utils.math_utils import Vector3


class Renderer:
    """OpenGL渲染器"""
    
    # 渲染模式常量
    MODE_SOLID = 'solid'
    MODE_WIREFRAME = 'wireframe'
    MODE_TRANSPARENT = 'transparent'
    MODE_TRANSPARENT_WIREFRAME = 'transparent_wireframe'
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "Skeletal Animation"):
        """
        初始化渲染器
        
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
        self.show_skeleton = True
        self.render_mode = self.MODE_TRANSPARENT_WIREFRAME
        self.background_color = (0.2, 0.2, 0.2, 1.0)
    
    def initialize(self) -> bool:
        """
        初始化OpenGL环境
        
        Returns:
            True 如果初始化成功
        """
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
        
        # 配置OpenGL
        self._setup_opengl()
        
        print(f"✓ OpenGL渲染器初始化成功")
        print(f"  版本: {glGetString(GL_VERSION).decode()}")
        
        return True
    
    def _setup_opengl(self):
        """配置OpenGL状态"""
        # 深度测试
        glEnable(GL_DEPTH_TEST)
        
        # 光照
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # 光源设置
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
    
    def render_frame(self, mesh: Mesh, deformer: Optional[SkinDeformer] = None, 
                     skeleton: Optional[Skeleton] = None):
        """
        渲染一帧
        
        Args:
            mesh: 网格模型
            deformer: 蒙皮变形器（可选）
            skeleton: 骨架（可选）
        """
        # 清空缓冲区
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置投影矩阵
        self._setup_projection()
        
        # 设置视图矩阵
        self._setup_view()
        
        # 渲染网格
        if deformer:
            self._render_deformed_mesh(mesh, deformer)
        else:
            self._render_mesh(mesh)
        
        # 渲染骨架
        if self.show_skeleton and skeleton:
            self._render_skeleton(skeleton)
        
        # 交换缓冲区
        glfw.swap_buffers(self.window)
    
    def _setup_projection(self):
        """设置投影矩阵"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = self.width / self.height
        gluPerspective(self.camera.fov, aspect, self.camera.near, self.camera.far)
    
    def _setup_view(self):
        """设置视图矩阵"""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 相机位置
        pos = self.camera.get_position()
        target = self.camera.target
        gluLookAt(pos.x, pos.y, pos.z,
                  target.x, target.y, target.z,
                  0, 0, 1)
        
        # 统一旋转场景（让模型站起来）
        glRotatef(90, 1, 0, 0)
    
    # ===== 网格渲染 =====
    
    def _render_mesh(self, mesh: Mesh):
        """渲染原始网格（未变形）"""
        glColor3f(0.8, 0.8, 0.8)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for i, idx in enumerate(face.vertex_indices):
                v = mesh.vertices[idx]
                
                # 法线
                if face.normal_indices and i < len(face.normal_indices):
                    n_idx = face.normal_indices[i]
                    if n_idx < len(mesh.normals):
                        n = mesh.normals[n_idx]
                        glNormal3f(n.x, n.y, n.z)
                
                glVertex3f(v.x, v.y, v.z)
        glEnd()
    
    def _render_deformed_mesh(self, mesh: Mesh, deformer: SkinDeformer):
        """
        渲染变形后的网格
        
        根据 render_mode 选择不同的渲染方式
        """
        vertices = deformer.get_deformed_vertices()
        normals = self._compute_normals(mesh, vertices)
        
        if self.render_mode == self.MODE_SOLID:
            self._draw_solid(mesh, vertices, normals)
        
        elif self.render_mode == self.MODE_WIREFRAME:
            self._draw_wireframe(mesh, vertices)
        
        elif self.render_mode == self.MODE_TRANSPARENT:
            self._draw_transparent(mesh, vertices, normals)
        
        else:  # MODE_TRANSPARENT_WIREFRAME
            self._draw_transparent_with_wireframe(mesh, vertices, normals)
    
    def _draw_solid(self, mesh: Mesh, vertices: List[Vector3], normals: List[Vector3]):
        """绘制实体网格"""
        glColor3f(0.7, 0.7, 0.7)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face.vertex_indices:
                glNormal3f(normals[idx].x, normals[idx].y, normals[idx].z)
                glVertex3f(vertices[idx].x, vertices[idx].y, vertices[idx].z)
        glEnd()
    
    def _draw_wireframe(self, mesh: Mesh, vertices: List[Vector3]):
        """绘制线框网格"""
        glDisable(GL_LIGHTING)
        glColor3f(0.0, 0.0, 0.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.0)
        
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face.vertex_indices:
                glVertex3f(vertices[idx].x, vertices[idx].y, vertices[idx].z)
        glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_LIGHTING)
    
    def _draw_transparent(self, mesh: Mesh, vertices: List[Vector3], normals: List[Vector3]):
        """绘制半透明网格"""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.8, 0.8, 0.8, 0.3)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for idx in face.vertex_indices:
                glNormal3f(normals[idx].x, normals[idx].y, normals[idx].z)
                glVertex3f(vertices[idx].x, vertices[idx].y, vertices[idx].z)
        glEnd()
        
        glDisable(GL_BLEND)
    
    def _draw_transparent_with_wireframe(self, mesh: Mesh, vertices: List[Vector3], 
                                         normals: List[Vector3]):
        """绘制半透明网格 + 线框"""
        # 先画半透明面
        self._draw_transparent(mesh, vertices, normals)
        
        # 再画黑色线框
        self._draw_wireframe(mesh, vertices)
    
    def _compute_normals(self, mesh: Mesh, vertices: List[Vector3]) -> List[Vector3]:
        """
        计算变形后顶点的法线
        
        方法：对每个顶点，平均其相邻面的法线
        
        Args:
            mesh: 原始网格
            vertices: 变形后的顶点列表
        
        Returns:
            法线列表
        """
        num_vertices = len(vertices)
        normals = [Vector3(0, 0, 0) for _ in range(num_vertices)]
        
        # 计算面法线并累加到顶点
        for face in mesh.faces:
            v0 = vertices[face.vertex_indices[0]]
            v1 = vertices[face.vertex_indices[1]]
            v2 = vertices[face.vertex_indices[2]]
            
            # 面法线 = 叉积
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = Vector3.cross(edge1, edge2)
            
            # 归一化
            length = face_normal.length()
            if length > 1e-8:
                face_normal = face_normal * (1.0 / length)
            
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
    
    # ===== 骨架渲染 =====
    
    def _render_skeleton(self, skeleton: Skeleton):
        """渲染骨架"""
        glDisable(GL_LIGHTING)
        
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
        
        glEnable(GL_LIGHTING)
    
    # ===== 事件处理 =====
    
    def should_close(self) -> bool:
        """检查窗口是否应该关闭"""
        return glfw.window_should_close(self.window)
    
    def poll_events(self):
        """处理窗口事件"""
        glfw.poll_events()
    
    def cleanup(self):
        """清理资源"""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("✓ 渲染器已清理")
