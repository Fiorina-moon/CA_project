"""
OBJ模型加载器
"""
from pathlib import Path
from typing import List

from src.core.mesh import Mesh, Face
from src.utils.math_utils import Vector3


class OBJLoader:
    """OBJ文件加载器"""
    
    @staticmethod
    def load(filepath: Path) -> Mesh:
        """
        加载OBJ文件
        
        Args:
            filepath: OBJ文件路径
        
        Returns:
            包含顶点、面、法线等数据的Mesh对象
        
        Note:
            - 支持顶点坐标（v）、纹理坐标（vt）、法线（vn）、面（f）
            - 如果文件中没有法线，会自动计算
            - OBJ索引从1开始，会自动转换为0开始
        """
        mesh = Mesh()
        mesh.name = filepath.stem
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                cmd = parts[0]
                
                # 顶点坐标
                if cmd == 'v':
                    x, y, z = map(float, parts[1:4])
                    mesh.vertices.append(Vector3(x, y, z))
                
                # 纹理坐标
                elif cmd == 'vt':
                    u, v = map(float, parts[1:3])
                    mesh.texcoords.append((u, v))
                
                # 法线
                elif cmd == 'vn':
                    nx, ny, nz = map(float, parts[1:4])
                    mesh.normals.append(Vector3(nx, ny, nz))
                
                # 面
                elif cmd == 'f':
                    face = OBJLoader._parse_face(parts[1:])
                    mesh.faces.append(face)
        
        # 如果没有法线，自动计算
        if not mesh.normals:
            mesh.compute_normals()
        
        # 输出加载信息
        print(f"✓ 加载网格: {mesh.name}")
        print(f"  顶点数: {mesh.get_vertex_count()}")
        print(f"  面数: {mesh.get_face_count()}")
        
        return mesh
    
    @staticmethod
    def _parse_face(face_parts: List[str]) -> Face:
        """
        解析面定义
        
        Args:
            face_parts: 面的顶点定义列表（例如 ['1/1/1', '2/2/2', '3/3/3']）
        
        Returns:
            Face对象
        
        支持的格式：
            - v           : 只有顶点索引
            - v/vt        : 顶点/纹理坐标
            - v/vt/vn     : 顶点/纹理坐标/法线
            - v//vn       : 顶点/法线（跳过纹理坐标）
        """
        vertex_indices = []
        texcoord_indices = []
        normal_indices = []
        
        for part in face_parts:
            indices = part.split('/')
            
            # 顶点索引（OBJ从1开始，转为0开始）
            vertex_indices.append(int(indices[0]) - 1)
            
            # 纹理坐标索引（可选）
            if len(indices) > 1 and indices[1]:
                texcoord_indices.append(int(indices[1]) - 1)
            
            # 法线索引（可选）
            if len(indices) > 2 and indices[2]:
                normal_indices.append(int(indices[2]) - 1)
        
        return Face(vertex_indices, normal_indices, texcoord_indices)
    
def load_obj(filepath) -> Mesh:
    """
    便捷函数：加载OBJ文件
    
    Args:
        filepath: OBJ文件路径（字符串或Path对象）
    
    Returns:
        Mesh对象
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    return OBJLoader.load(filepath)