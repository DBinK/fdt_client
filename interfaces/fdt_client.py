# pytracking_client.py
import json
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

    def _encode_frame(self, frame: np.ndarray):
        """编码图像为JPEG（二进制）或根据数据类型选择适当编码"""
        if frame.dtype == np.uint16:    
            # _, buf = cv2.imencode(".png", frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # return buf.tobytes()  # 对于uint16深度图, 使用PNG进行无损压缩，保留更多细节
            return frame.tobytes()  # 对于uint16深度图, 直接返回原始数据以保证精度和最小延迟
        else:
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes()  # 对于普通彩色图像，使用JPEG

    def init(self, color_frame: np.ndarray, depth_frame: np.ndarray, object_name: str, cam_K: np.ndarray):
        logger.info(f"{color_frame.shape:}, {object_name:}, {cam_K:}")

        meta = {"cmd": "init", "object_name": object_name, "cam_K": cam_K, "session_id": self.session_id}
        color_frame_bytes = self._encode_frame(color_frame)
        depth_frame_bytes = self._encode_frame(depth_frame)

        # multipart: [json, binary]
        self.zmq_socket.send_multipart([json.dumps(meta).encode("utf-8"), color_frame_bytes, depth_frame_bytes])
        reply = self.zmq_socket.recv_json()  # 接收回复

        # 确保reply是字典类型
        if isinstance(reply, dict) and reply.get("status") == "ok":
            self.initialized = True
            logger.success(f"tracker 初始化成功: {reply}")
            return True
        else:
            logger.error(f"tracker 初始化失败: {reply}")
            return False

    def update(self, color_frame: np.ndarray, depth_frame: np.ndarray):
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call init() first.")

        meta = {"cmd": "update", "session_id": self.session_id}

        color_frame_bytes = self._encode_frame(color_frame)
        depth_frame_bytes = self._encode_frame(depth_frame)

        self.zmq_socket.send_multipart([json.dumps(meta).encode("utf-8"), color_frame_bytes, depth_frame_bytes])
        reply = self.zmq_socket.recv_json()  # 接收回复
        
        return reply

        # TODO: 等待确定返回值
        # # 确保reply是字典类型
        # if isinstance(reply, dict) and reply.get("status") == "ok":
        #     bbox = reply.get("bbox")  # 确保bbox可以转换为tuple类型
        #     if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        #         logger.debug(f"任务 {self.session_id} 更新的 bbox {bbox}")
        #         return True, tuple(bbox)  
        #     else:              
        #         logger.error(f"任务 {self.session_id} bbox 格式错误: {bbox}")
        #         return False, None   # 如果bbox不是期望的格式，则返回错误
        # else:
        #     return False, None

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


if __name__ == "__main__":
    
    from rich import print as rprint
    from rgbd_cam import OrbbecRGBDCamera

    tracker = RemoteFoundationPose()
    tracker.release()

    # 创建采集器实例
    gemini_2 = OrbbecRGBDCamera(device_index=0)

    loop = LoopTick()

    is_recording = False

    try:
        # 启动采集
        gemini_2.start()

        intrinsic = gemini_2.get_intrinsic()
        distortion = gemini_2.get_depth_distortion()

        rprint(intrinsic)
        rprint(distortion)

        breakpoint()
        
        print("开始采集彩色和深度图像，按 'q' 或 ESC 键退出")
        
        while True:
            # 获取图像帧
            color_image, depth_image, depth_data = gemini_2.get_frames()
            ns = loop.tick()
            hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)

            # 显示彩色图像
            if color_image is None or depth_image is None:
                continue

            cv2.imshow("Color Image", color_image)
                
            # 显示深度图像
            depth_image = cv2.bitwise_not(depth_image)
            cv2.imshow("Depth Image", depth_image)

            if not tracker.initialized and intrinsic is not None:
                tracker.init(color_image, depth_image, "mango", intrinsic)

            reply = tracker.update(color_image, depth_image)

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