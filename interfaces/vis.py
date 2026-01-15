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
    
    # 确保 pose 是 4x4 矩阵
    if isinstance(pose, (list, tuple)):
        pose = np.array(pose).reshape(4, 4)
    elif isinstance(pose, np.ndarray) and pose.size == 16:
        pose = pose.reshape(4, 4)
    
    # 确保 K 是 3x3 矩阵
    if isinstance(K, (list, tuple)):
        K = np.array(K).reshape(3, 3)
    elif isinstance(K, np.ndarray) and K.size == 9:
        K = K.reshape(3, 3)
    
    # 确保 bbox_3d_local 是 numpy 数组
    if not isinstance(bbox_3d_local, np.ndarray):
        bbox_3d_local = np.array(bbox_3d_local)

    # 1. 将局部坐标转换到相机坐标: P_cam = Pose * P_local
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
        (0, 1), (1, 2), (2, 3), (3, 0), # 底面
        (4, 5), (5, 6), (6, 7), (7, 4), # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 垂直连接
    ]
    
    for i, j in lines:
        pt1 = tuple(uv[i])
        pt2 = tuple(uv[j])
        # 简单裁剪，避免画在屏幕外报错
        cv2.line(img_draw, pt1, pt2, (0, 255, 200), 1, cv2.LINE_AA)
        
    # 计算并绘制立方体中心点
    # 立方体中心 = 所有顶点的平均值
    cube_center_3d = np.mean(bbox_3d_local, axis=0)  # (3,) - 局部坐标系中的中心点
    
    # 将立方体中心转换到相机坐标系
    ones = np.array([*cube_center_3d, 1])  # 齐次坐标
    cube_center_cam = (pose @ ones)[:3]  # 相机坐标系中的中心点
    
    # 将立方体中心投影到图像平面
    if cube_center_cam[2] > 0:  # 确保点在相机前方
        center_uv = K @ cube_center_cam
        cx, cy = int(center_uv[0]/center_uv[2]), int(center_uv[1]/center_uv[2])
        cv2.circle(img_draw, (cx, cy), 6, (255, 0, 0), -1)  # 使用蓝色圆圈标记立方体中心
        
        # 绘制坐标轴（从立方体中心开始）
        axis_length = 0.1  # 坐标轴长度
        
        # 使用旋转矩阵的列向量来定义各轴方向（从立方体中心出发）
        rotation_matrix = pose[:3, :3]
        
        # X轴（红色）在相机坐标系中的终点（从立方体中心出发）
        x_end_cam = cube_center_cam + rotation_matrix[:3, 0] * axis_length
        # Y轴（绿色）在相机坐标系中的终点（从立方体中心出发）
        y_end_cam = cube_center_cam + rotation_matrix[:3, 1] * axis_length
        # Z轴（蓝色）在相机坐标系中的终点（从立方体中心出发）
        z_end_cam = cube_center_cam + rotation_matrix[:3, 2] * axis_length
        
        # 投影到图像平面
        def project_point(point_cam):
            if point_cam[2] > 0:
                point_uv = K @ point_cam
                u, v = int(point_uv[0]/point_uv[2]), int(point_uv[1]/point_uv[2])
                return (u, v)
            return None
        
        # 投影立方体中心和各轴端点
        center_proj = (cx, cy)  # 已经计算好的立方体中心
        x_proj = project_point(x_end_cam)
        y_proj = project_point(y_end_cam)
        z_proj = project_point(z_end_cam)
        
        # 绘制坐标轴（从立方体中心点开始）
        if center_proj and x_proj:
            cv2.line(img_draw, center_proj, x_proj, (0, 0, 255), 3)  # X轴 - 红色
        if center_proj and y_proj:
            cv2.line(img_draw, center_proj, y_proj, (0, 255, 0), 3)  # Y轴 - 绿色
        if center_proj and z_proj:
            cv2.line(img_draw, center_proj, z_proj, (255, 0, 0), 3)  # Z轴 - 蓝色


    return img_draw