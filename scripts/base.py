from fdt_client import track_pose


if __name__ == "__main__":
    text_prompt = "yellow"
    mesh_file = "tmp/scaled_mesh.obj"

    track_pose(
        text_prompt=text_prompt,
        mesh_file=mesh_file,
        device_index=1,
        server_url="tcp://127.0.0.1:5555",
        vis=False,
    )
