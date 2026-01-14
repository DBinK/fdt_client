# fdt_server.py
import json
import threading
import time
import shutil
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import zmq
import numpy as np
from logger import logger
from rich import print as rprint

# === 引入新的 Tracker ===
from tracker import FoundationPoseGDSAMTracker

logger.remove()
logger.add(sys.stderr, level="INFO")

# 临时文件存储路径
TEMP_DIR = Path("tmp_sessions")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class SessionThread(threading.Thread):
    def __init__(self, session_id, color_frame, depth_frame, text_prompt, mesh_path, cam_K, vis=False):
        super().__init__()
        self.session_id = session_id
        self.color_frame = color_frame
        self.depth_frame = depth_frame
        self.text_prompt = text_prompt
        self.mesh_path = mesh_path  # 这是本地保存的临时文件路径
        self.cam_K = cam_K
        self.vis = vis
        
        self.tracker = None
        self.initialized = False
        self.stop_event = threading.Event()
        
        # 结果传递
        self.result = None
        self.result_available = threading.Event()
        
        # 调试目录
        self.debug_dir = TEMP_DIR / session_id / "debug"

    def run(self):
        try:
            logger.info(f"会话 {self.session_id} 初始化 FoundationPose...")
            
            # 初始化跟踪器对象
            self.tracker = FoundationPoseGDSAMTracker(
                text_prompt=self.text_prompt,
                mesh_file=str(self.mesh_path),
                K=self.cam_K,
                show_vis=False,     # 服务端不弹窗，太慢且无法传输
                save_vis=self.vis,  # 如果开启vis，则保存图片到debug目录
                save_3d=False,
                debug_dir=str(self.debug_dir),
                device="cuda"
            )
            logger.info(f"会话 {self.session_id} 追踪器初始化完成，开始第一帧注册...")
            
            # 执行第一帧注册
            success, pose = self.tracker.init(self.color_frame, self.depth_frame)
            logger.info(f"会话 {self.session_id} 第一帧注册完成，结果: {success}")
            
            
            if success:
                rprint(pose)  # 打印 Pose
                # 获取 BBox (8个顶点) 供客户端画图
                bbox_corners = self.tracker.get_bbox_corners()
                
                self.initialized = True
                self.result = {
                    "status": "ok",
                    "pose": pose.flatten().tolist(),     # 转换为列表 (16,)
                    "bbox": bbox_corners.tolist()        # 转换为列表 (8, 3)
                }
            else:
                self.result = {"status": "error", "msg": "Object not detected in first frame"}

        except Exception as e:
            logger.error(f"会话 {self.session_id} 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.result = {"status": "error", "msg": str(e)}
        finally:
            self.result_available.set()
            
            # 保持线程运行以处理后续请求（实际上这个线程模型里，run跑完init就没事做了，
            # 只要 tracker 实例还在内存里，主线程可以直接调用 tracker.update）
            # 为了保持对象存活，我们在主类里持有 session_thread 实例
            pass

    def update(self, color_frame, depth_frame):
        """主线程调用的同步更新方法"""
        if not self.initialized or self.tracker is None:
            return None

        try:
            success, pose = self.tracker.update(color_frame, depth_frame)
            
            if success:
                # 返回 Pose (4x4 -> flattened list)
                return pose.flatten().tolist()
            else:
                return None
            
        except Exception as e:
            logger.error(f"会话 {self.session_id} 更新失败: {e}")
            return None

    def stop(self):
        self.stop_event.set()
        # 清理临时文件
        try:
            session_dir = TEMP_DIR / self.session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info(f"清理会话临时文件: {session_dir}")
        except Exception as e:
            logger.error(f"清理文件失败: {e}")


class TrackerServer:
    def __init__(self, vis=False, address="tcp://*:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.address = address
        self.socket.bind(self.address)
        
        # 存储多个会话线程，以会话ID为键
        self.sessions = {}
        self.sessions_lock = threading.Lock()
        self.vis = vis
        
        logger.info("服务端启动...")
        logger.info(f"监听地址: {self.address}")

    def decode_frame(self, buf, dtype=np.uint8, shape=None):
        """
        :param shape: tuple (height, width)，用于 raw bytes 重塑
        """
        if buf is None or len(buf) == 0:
            return None
            
        if dtype == np.uint16:
            # 1. 优先尝试 PNG 解码 (兼容性)
            try:
                arr = np.frombuffer(buf, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # 将uint16图像转换为float32以避免PyTorch错误
                    if img.dtype == np.uint16:
                        img = img.astype(np.float32)
                    return img
            except:
                pass
            
            # 2. 回退到 Raw Bytes 解析
            # 如果提供了 shape，则使用 shape；否则报错或使用默认值
            raw_data = np.frombuffer(buf, dtype=np.uint16)
            
            if shape is not None:
                h, w = shape
                # 安全检查：确保字节数匹配
                if raw_data.size != h * w:
                    logger.error(f"Raw data size {raw_data.size} does not match shape {shape} ({h*w})")
                    return None
                # 将uint16转换为float32以避免PyTorch错误
                return raw_data.reshape((h, w)).astype(np.float32)
            else:
                # 只有在万不得已时才硬编码
                logger.warning("Warning: Decoding raw depth without shape! Assuming 480x640.")
                return raw_data.reshape((1280, 720))  # 一般都是这个分辨率
                
        else:
            # 彩色图 (JPG) 自带分辨率，无需 shape
            np_arr = np.frombuffer(buf, dtype=dtype)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def save_temp_mesh(self, session_id, mesh_bytes):
        """将接收到的 mesh 二进制写入临时文件"""
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = session_dir / "uploaded_mesh.obj"
        with open(mesh_path, "wb") as f:
            f.write(mesh_bytes)
        return mesh_path

    def handle_init_command(self, parts):
        """处理 Init: [json, mesh_bytes, color_bytes, depth_bytes]"""
        if len(parts) != 4:
            return {"status": "error", "msg": "Init requires 4 parts: [meta, mesh, color, depth]"}

        meta_str, mesh_buf, color_buf, depth_buf = parts
        try:
            msg = json.loads(meta_str.decode("utf-8"))
        except Exception as e:
            return {"status": "error", "msg": "invalid json " + str(e)}

        session_id = msg.get("session_id")
        text_prompt = msg.get("text_prompt")
        cam_K_list = msg.get("cam_K")
        # 从 JSON 获取分辨率
        width = msg.get("width", 1280)  # 提供默认值以防老客户端连接
        height = msg.get("height", 720)
        target_shape = (height, width) # 注意 numpy 是 (h, w)

        if not all([session_id, text_prompt, cam_K_list]):
            return {"status": "error", "msg": "Missing required fields"}

        # 解码图像
        color_frame = self.decode_frame(color_buf, np.uint8)
        depth_frame = self.decode_frame(depth_buf, np.uint16, shape=target_shape)
        
        if color_frame is None or depth_frame is None:
            return {"status": "error", "msg": "Frame decode failed"}

        # 保存 Mesh
        try:
            mesh_path = self.save_temp_mesh(session_id, mesh_buf)
        except Exception as e:
            return {"status": "error", "msg": f"Save mesh failed: {e}"}

        # 启动会话
        cam_K = np.array(cam_K_list) # 转换为 numpy
        
        with self.sessions_lock:
            if session_id in self.sessions:
                self.sessions[session_id].stop()
            
            session_thread = SessionThread(
                session_id, color_frame, depth_frame, 
                text_prompt, mesh_path, cam_K, self.vis
            )
            session_thread.start()
            self.sessions[session_id] = session_thread
            
            # 等待初始化完成
            session_thread.result_available.wait()
            return session_thread.result

    def handle_update_command(self, parts): 
        """处理 Update: [json, color_bytes, depth_bytes]"""
        if len(parts) != 3:
            return {"status": "error", "msg": "Update requires 3 parts"}
            
        meta_str, color_buf, depth_buf = parts
        try:
            msg = json.loads(meta_str.decode("utf-8"))
        except Exception as e:
            return {"status": "error", "msg": "invalid json" + str(e)}
            
        session_id = msg.get("session_id")
        
        with self.sessions_lock:
            if session_id not in self.sessions:
                return {"status": "error", "msg": "Session not found"}
            session_thread = self.sessions[session_id]

        if not session_thread.initialized:
            return {"status": "error", "msg": "Tracker not initialized"}

        # 从 JSON 获取分辨率
        width = msg.get("width", 1280)  # 提供默认值以防老客户端连接
        height = msg.get("height", 720)
        target_shape = (height, width) # 注意 numpy 是 (h, w)

        # 解码图像
        color_frame = self.decode_frame(color_buf, np.uint8)
        depth_frame = self.decode_frame(depth_buf, np.uint16, shape=target_shape) # 修正：添加shape参数

        pose_list = session_thread.update(color_frame, depth_frame)
        
        if pose_list is not None:
            return {"status": "ok", "pose": pose_list}
        else:
            return {"status": "error", "msg": "Tracking lost"}

    def handle_release_command(self, msg):
        session_id = msg.get("session_id")
        with self.sessions_lock:
            if session_id in self.sessions:
                self.sessions[session_id].stop()
                del self.sessions[session_id]
                return {"status": "ok"}
        return {"status": "error", "msg": "Session not found"}

    def run(self):
        try:
            while True:
                # 接收 multipart
                parts = self.socket.recv_multipart()
                
                # 先解析第一部分 JSON 确定命令类型
                try:
                    meta_json = json.loads(parts[0].decode("utf-8"))
                    cmd = meta_json.get("cmd")
                except Exception as e:
                    logger.error(f"JSON Parse Error: {e}")
                    self.socket.send_json({"status": "error", "msg": "Invalid metadata"})
                    continue

                logger.debug(f"收到命令: {cmd}, Parts数量: {len(parts)}")

                response = {"status": "error", "msg": "Unknown command"}

                if cmd == "init":
                    response = self.handle_init_command(parts)
                elif cmd == "update":
                    response = self.handle_update_command(parts)
                elif cmd == "release":
                    response = self.handle_release_command(meta_json)
                
                self.socket.send_json(response)

        except Exception as e:
            logger.error(f"Server loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self.sessions_lock:
                for s in self.sessions.values():
                    s.stop()

if __name__ == "__main__":
    # 确保 config.py 和 weights 路径正确
    server = TrackerServer(vis=True) # vis=True 会在服务器端 debug 目录保存图片
    server.run()