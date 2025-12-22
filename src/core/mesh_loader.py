"""
OBJ模型加载器
"""
from pathlib import Path
from typing import List, Tuple
from core.mesh import Mesh, Face
from utils.math_utils import Vector3


class OBJLoader:
    """OBJ文件加载器"""
    
    @staticmethod
    def load(filepath: Path) -> Mesh:
        """
        加载OBJ文件
        
        Args:
            filepath: OBJ文件路径
        
        Returns:
            Mesh对象
        """
        mesh = Mesh()
        mesh.name = filepath.stem
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # 顶点坐标
                if parts[0] == 'v':
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    mesh.vertices.append(Vector3(x, y, z))
                
                # 纹理坐标
                elif parts[0] == 'vt':
                    u, v = float(parts[1]), float(parts[2])
                    mesh.texcoords.append((u, v))
                
                # 法线
                elif parts[0] == 'vn':
                    nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                    mesh.normals.append(Vector3(nx, ny, nz))
                
                # 面
                elif parts[0] == 'f':
                    face = OBJLoader._parse_face(parts[1:])
                    mesh.faces.append(face)
        
        # 如果没有法线，计算法线
        if not mesh.normals:
            mesh.compute_normals()
        
        print(f"✓ Loaded mesh: {mesh.name}")
        print(f"  Vertices: {mesh.get_vertex_count()}")
        print(f"  Faces: {mesh.get_face_count()}")
        
        return mesh
    
    @staticmethod
    def _parse_face(face_parts: List[str]) -> Face:
        """
        解析面定义
        格式: v, v/vt, v/vt/vn, v//vn
        """
        vertex_indices = []
        texcoord_indices = []
        normal_indices = []
        
        for part in face_parts:
            indices = part.split('/')
            
            # 顶点索引（OBJ索引从1开始，转为0）
            vertex_indices.append(int(indices[0]) - 1)
            
            # 纹理坐标索引
            if len(indices) > 1 and indices[1]:
                texcoord_indices.append(int(indices[1]) - 1)
            
            # 法线索引
            if len(indices) > 2 and indices[2]:
                normal_indices.append(int(indices[2]) - 1)
        
        return Face(vertex_indices, normal_indices, texcoord_indices)