import numpy as np
import cv2

def draw_3d_box_client(image, pose, K, bbox_3d_local):
    """
    在客户端快速绘制 3D 框
    :param image: 当前帧图像 (H, W, 3)
    :param pose: 4x4 位姿矩阵 (物体 -> 相机)
    :param K: 3x3 相机内参
    :param bbox_3d_local: (8, 3) 物体局部坐标系下的8个顶点 (由服务端 init 时传回)
    """
    img_draw = image.copy()
    
    # 1. 将局部坐标转换到相机坐标: P_cam = Pose * P_local
    # bbox_3d_local 是 (8, 3)，变成齐次坐标 (8, 4)
    ones = np.ones((bbox_3d_local.shape[0], 1))
    points_homo = np.hstack([bbox_3d_local, ones]) # (8, 4)
    
    # 矩阵乘法: (Pose @ points.T).T -> (8, 4)
    points_cam = (pose @ points_homo.T).T 
    
    # 取出前3维 (X, Y, Z)
    xyz = points_cam[:, :3]
    
    # 2. 投影到 2D 像素平面: p_2d = K * P_cam / Z
    # 矩阵乘法: (K @ xyz.T).T -> (8, 3)
    uv_z = (K @ xyz.T).T
    
    # 归一化 (u/z, v/z)
    z = uv_z[:, 2:] + 1e-6 # 避免除以0
    uv = uv_z[:, :2] / z
    
    # 转整数像素坐标
    uv = uv.astype(np.int32)
    
    # 3. 连线 (定义立方体的12条棱)
    # 假设 bbox 顺序是 trimesh.bounds.corners 的标准顺序
    # 也可以简单粗暴地根据距离画，这里给出一个通用的连接表
    lines = [
        (0, 1), (1, 3), (3, 2), (2, 0), # 底面
        (4, 5), (5, 7), (7, 6), (6, 4), # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 中间柱子
    ]
    
    for i, j in lines:
        pt1 = tuple(uv[i])
        pt2 = tuple(uv[j])
        # 简单裁剪，避免画在屏幕外报错
        cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        
    # 画个坐标轴中心看看位置对不对
    center_cam = pose[:3, 3]
    center_uv = (K @ center_cam)
    if center_uv[2] > 0:
        cx, cy = int(center_uv[0]/center_uv[2]), int(center_uv[1]/center_uv[2])
        cv2.circle(img_draw, (cx, cy), 5, (0, 0, 255), -1)

    return img_draw