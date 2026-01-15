# main.py

from pathlib import Path
import cv2
from looptick import LoopTick
from rich import print as rprint

from fdt_client.rgbd_cam import OrbbecRGBDCamera
from fdt_client.client import RemoteFoundationPose
from fdt_client.vis import draw_3d_box_client


loop = LoopTick()

def track_pose(
    text_prompt: str,
    mesh_file: Path | str,
    device_index: int = 1,
    server_url: str = "tcp://127.0.0.1:5555",
    vis: bool = False
):
    tracker = RemoteFoundationPose(server_url)
    gemini_2 = OrbbecRGBDCamera(device_index=device_index)
    
    try:
        # 启动采集
        gemini_2.start()

        mesh_file = Path(mesh_file)
        intrinsic = gemini_2.get_intrinsic()

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

            print(f"\rHz: {hz:.2f}", end='', flush=True)

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
            
            if not is_init:
                continue

            ret, pose = tracker.update(color_image, depth_image)

            if ret and vis:
                color_image = draw_3d_box_client(color_image, pose, intrinsic, tracker.bbox_corners)
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

def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description='Run FoundationPose tracking with Orbbec RGB-D camera.')
    parser.add_argument('--text-prompt', type=str, default="yellow", help='Text prompt for object detection (default: yellow)')
    parser.add_argument('--mesh-file', type=str, default="tmp/scaled_mesh.obj", help='Path to the mesh file (default: tmp/scaled_mesh.obj)')
    parser.add_argument('--device-index', type=int, default=1, help='Camera device index (default: 1)')
    parser.add_argument('--server-url', type=str, default="tcp://127.0.0.1:5555", help='FoundationPose server URL (default: tcp://127.0.0.1:5555)')
    
    args = parser.parse_args()
    
    track_pose(args.text_prompt, args.mesh_file, args.device_index, args.server_url)


if __name__ == "__main__":

    # run_cli()

    text_prompt = "yellow"
    mesh_file = "tmp/scaled_mesh.obj"

    track_pose(text_prompt, mesh_file,vis=True)
"""
python src/fdt_client/main.py --text-prompt yellow --mesh-file tmp/scaled_mesh.obj
"""