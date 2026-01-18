# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path

block_cipher = None

# ========== 查找关键 DLL ==========
def find_glfw_dll():
    """查找 GLFW DLL"""
    try:
        import glfw
        glfw_dir = Path(glfw.__file__).parent
        for dll_name in ['glfw3.dll', 'glfw.dll']:
            dll_path = glfw_dir / dll_name
            if dll_path.exists():
                print(f"[找到 GLFW] {dll_path}")
                return (str(dll_path), '.')
    except Exception as e:
        print(f"[警告] 未找到 GLFW DLL: {e}")
    return None

def find_opencv_dlls():
    """查找 OpenCV DLL"""
    try:
        import cv2
        cv2_dir = Path(cv2.__file__).parent
        dlls = list(cv2_dir.glob('*.dll'))
        if dlls:
            print(f"[找到 OpenCV] {len(dlls)} 个 DLL")
            return [(str(dll), '.') for dll in dlls]
    except Exception as e:
        print(f"[警告] 未找到 OpenCV DLL: {e}")
    return []

# 收集所有 DLL
binaries = []

glfw_dll = find_glfw_dll()
if glfw_dll:
    binaries.append(glfw_dll)

opencv_dlls = find_opencv_dlls()
if opencv_dlls:
    binaries.extend(opencv_dlls)

# ========== Analysis 配置 ==========
a = Analysis(
    ['ui_main.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('data', 'data'),  # 模型和动画数据
        ('src', 'src'),    # 源代码
    ],
    hiddenimports=[
        # Qt5
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtOpenGL',
        'PyQt5.sip',
        
        # OpenGL
        'OpenGL',
        'OpenGL.GL',
        'OpenGL.GLU',
        'OpenGL.GLUT',
        'OpenGL.arrays',
        'OpenGL.platform',
        
        # 窗口库
        'glfw',
        
        # 数值计算
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'scipy',
        'scipy.sparse',
        'scipy.sparse.csgraph',
        'scipy.spatial',
        
        # 图像处理
        'PIL',
        'PIL.Image',
        'cv2',
        'imageio',
        'imageio.plugins',
        
        # 绘图
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        
        # 3D 处理
        'trimesh',
        'pygltflib',
        
        # 工具库
        'dataclasses',
        'dataclasses_json',
        'tqdm',
        'colorama',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',      # 不使用 Tkinter
        'matplotlib.tests',
        'scipy.tests',
        'numpy.tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ========== PYZ 配置 ==========
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ========== EXE 配置 ==========
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CA_Animation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 调试时设为 True，发布时改为 False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可选：添加图标 'icon.ico'
)

# ========== COLLECT 配置 ==========
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CA_Animation'
)
