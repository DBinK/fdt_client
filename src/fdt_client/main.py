# main.py

import cv2
from looptick import LoopTick
from rich import print as rprint

from rgbd_cam import OrbbecRGBDCamera
from client import RemoteFoundationPose
from vis import draw_3d_box_client

# 测试用例
if __name__ == "__main__":
    


    tracker = RemoteFoundationPose("tcp://127.0.0.1:5555")
    tracker.release()

    # 创建采集器实例
    gemini_2 = OrbbecRGBDCamera(device_index=1)

    loop = LoopTick()

    is_recording = False

    try:
        # 启动采集
        gemini_2.start()

        # text_prompt = "mango"
        text_prompt = "yellow"
        intrinsic = gemini_2.get_intrinsic()
        mesh_file = "tmp/scaled_mesh.obj"

        rprint(intrinsic)
        
        print("开始采集彩色和深度图像，按 'q' 或 ESC 键退出")
        is_init = False
        while True:
            # 获取图像帧
            color_image, depth_image, depth_data = gemini_2.get_frames()

            # 非空检查 (防止运行时报错 -215 Assertion failed)
            if color_image is None or depth_image is None:
                print("图像帧为空，请检查设备是否正常连接")
                continue

            ns = loop.tick()
            hz = 1 / ((ns * loop.NS2SEC) if ns > 0.01 else 0.01)
            print(hz)

            # 显示彩色图像 和 深度图像
            # depth_image_show = cv2.bitwise_not(depth_image)
            # cv2.imshow("Color Image", color_image)
            # cv2.imshow("Depth Image", depth_image_show)

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
                ret, pose = tracker.update(color_image, depth_image)

                if ret:
                    bbox = tracker.bbox_corners
                    # 绘制3D框
                    color_image = draw_3d_box_client(color_image, pose, intrinsic, bbox)
                    cv2.imshow("Color Image", color_image)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                break
                    
    except Exception as e:
        print(f"程序运行出错: {e}")
    except KeyboardInterrupt:
        print("用户中断程序")
    finally:
        cv2.destroyAllWindows()
        gemini_2.stop()
        tracker.release()