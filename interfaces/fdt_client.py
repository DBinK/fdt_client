# pytracking_client.py
import json
import os
from pathlib import Path
import uuid
import sys

from looptick import LoopTick
import numpy as np
import zmq
import cv2
from loguru import logger   # 设置日志级别为DEBUG，这样就能看到debug信息了


logger.remove()  # 移除默认的处理器
logger.add(sys.stderr, level="INFO")  # 添加新的处理器，级别为DEBUG
# logger.add(sys.stderr, level="DEBUG")  # 添加新的处理器，级别为DEBUG


class RemoteFoundationPose:
    def __init__(self, address="tcp://127.0.0.1:5555"):
        self.context = zmq.Context()
        self.zmq_socket = self.context.socket(zmq.REQ)
        self.zmq_socket.connect(address)
        self.initialized = False
        self.session_id = str(uuid.uuid4())  # 在客户端生成会话ID

        logger.info(f"连接到 {address}")
        logger.info(f"任务 Session ID: {self.session_id}")

        self.init_pose = []
        self.bbox_corners = []

    def _encode_frame(self, frame: np.ndarray):
        """编码图像为JPEG（二进制）或根据数据类型选择适当编码"""
        if frame.dtype == np.uint16:    
            # _, buf = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # return buf.tobytes()  # 对于uint16深度图, 使用PNG进行无损压缩，保留更多细节
            return frame.tobytes()  # 对于uint16深度图, 直接返回原始数据以保证精度和最小延迟
        else:
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes()  # 对于普通彩色图像，使用JPEG

    def _encode_file(self, file_path:Path | str):
        """编码文件为二进制数据"""
        file_path = Path(file_path)
        if file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return f.read()
        else:
            logger.warning(f"文件不存在: {file_path}")
            return b""

    def init(
        self,
        text_prompt: str,
        cam_K: np.ndarray,
        mesh_file: Path | str,
        color_frame: np.ndarray,
        depth_frame: np.ndarray,
    ):
        logger.info(f"{color_frame.shape:}, {text_prompt:}, {cam_K:}")

        h, w = depth_frame.shape[:2]

        meta = {
            "cmd": "init",
            "session_id": self.session_id,
            "width": w,   # 图像宽高, 服务端解析图片时需要
            "height": h,   
            "text_prompt": text_prompt,
            "cam_K": cam_K.tolist(),
        }
        color_frame_bytes = self._encode_frame(color_frame)
        depth_frame_bytes = self._encode_frame(depth_frame)
        mesh_file_bytes = self._encode_file(mesh_file)

        # multipart: [json, binary]
        self.zmq_socket.send_multipart([
                json.dumps(meta).encode("utf-8"), 
                mesh_file_bytes,
                color_frame_bytes, 
                depth_frame_bytes,
            ])
        reply = self.zmq_socket.recv_json()  # 接收回复

        # 确保reply是字典类型
        if isinstance(reply, dict) and reply.get("status") == "ok":
            self.initialized = True
            logger.success(f"tracker 初始化成功: {reply}")
            self.init_pose = reply.get("pose")  # 在客户端其实没什么用
            self.bbox_corners = reply.get("bbox")
            return True
        else:
            logger.error(f"tracker 初始化失败: {reply}")
            return False

    def update(self, color_frame: np.ndarray, depth_frame: np.ndarray):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")
        
        h, w = depth_frame.shape[:2]

        meta = {
            "cmd": "update",
            "session_id": self.session_id,
            "width": w,  # 图像宽高, 服务端解析图片时需要
            "height": h,
        }

        color_frame_bytes = self._encode_frame(color_frame)
        depth_frame_bytes = self._encode_frame(depth_frame)

        self.zmq_socket.send_multipart([json.dumps(meta).encode("utf-8"), color_frame_bytes, depth_frame_bytes])
        reply = self.zmq_socket.recv_json()  # 接收回复
        
        # return reply

        # TODO: 等待确定返回值
        # # 确保reply是字典类型
        if isinstance(reply, dict) and reply.get("status") == "ok":
            pose = reply.get("pose")   # 确保bbox可以转换为tuple类型
            if isinstance(pose, (list, tuple)) and len(pose) == 16:
                logger.debug(f"任务 {self.session_id} 更新的 pose {pose}")
                return True, tuple(pose)  
            else:              
                logger.error(f"任务 {self.session_id} pose 格式错误: {pose}")
                return False, None   # 如果bbox不是期望的格式，则返回错误
        else:
            return False, None

    def release(self):
        """
        释放远程跟踪器资源
        """
        meta = {"cmd": "release", "session_id": self.session_id}
        self.zmq_socket.send_json(meta)
        reply = self.zmq_socket.recv_json()
        
        if isinstance(reply, dict) and reply.get("status") == "ok":
            self.initialized = False
            logger.info("tracker 已释放")
            return True
        else:
            logger.error("tracker 释放失败")
            return False    


# 测试用例
if __name__ == "__main__":
    
    from rich import print as rprint
    from rgbd_cam import OrbbecRGBDCamera


    tracker = RemoteFoundationPose("tcp://127.0.0.1:5555")
    tracker.release()

    # 创建采集器实例
    gemini_2 = OrbbecRGBDCamera(device_index=0)

    loop = LoopTick()

    is_recording = False

    try:
        # 启动采集
        gemini_2.start()

        text_prompt = "yellow"
        intrinsic = gemini_2.get_intrinsic()
        mesh_file = "tmp/scaled_mesh.obj"

        rprint(intrinsic)
        
        print("开始采集彩色和深度图像，按 'q' 或 ESC 键退出")
        is_init = False
        while True:
            # 获取图像帧
            # color_raw, _, depth_raw = gemini_2.get_frames()

            # 1. 获取数据
            color_image, depth_image, depth_data = gemini_2.get_frames()

            
            # 2. 关键：非空检查 (防止运行时报错 -215 Assertion failed)
            if color_image is None or depth_image is None:
                print("图像帧为空，请检查设备是否正常连接")
                continue

            # # 3. 关键：强制类型转换 (解决 Pylance 报错)
            # # 即使 color_raw 已经是 array，这一步也能消除编辑器的类型疑虑
            # # 同时也兼容了部分 SDK 返回的是自定义 Frame 对象的情况
            # color_np = np.asanyarray(color_image)
            # depth_np = np.asanyarray(depth_data)

            # # 4. 颜色空间转换 (RGB -> BGR)
            # # 现在的 color_np 肯定是合法的 numpy 数组了
            # color_image = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # # 5. 深度数据转换
            # depth_image = depth_np.astype(np.uint16)


            # color_path = "/home/zyh/wmz/our_data/processed/mangguo/1764919025923146/rgb/000001.png"
            # depth_path = "/home/zyh/wmz/our_data/processed/mangguo/1764919025923146/depth/000001.png"
            # # 读取为 BGR 格式
            # color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)

            # # 关键：使用 IMREAD_UNCHANGED 保持 16 位精度
            # depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # # 转换为米 (Meters)
            # # 假设本地 PNG 的单位是毫米 (mm)，这是最常见的情况
            # depth_image = depth_uint16.astype(np.float32)


            if color_image is None or depth_data is None:
                continue

            ns = loop.tick()
            hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)

            # depth_image = cv2.bitwise_not(depth_image)
            # 显示彩色图像 和 深度图像
            # cv2.imshow("Color Image", color_image)
            # cv2.imshow("Depth Image", depth_image)

            if not tracker.initialized and intrinsic is not None:
                is_init = tracker.init(
                    text_prompt=text_prompt,
                    cam_K=intrinsic,
                    mesh_file=mesh_file,
                    color_frame=color_image,
                    depth_frame=depth_image,
                )

            # breakpoint()
            
            if is_init:
                reply = tracker.update(color_image, depth_image)
                rprint(reply)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                break
                    
    except Exception as e:
        print(f"程序运行出错: {e}")
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        gemini_2.stop()
        cv2.destroyAllWindows()