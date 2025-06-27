import bpy
import sys
import os
import numpy as np
import cv2

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if not argv:
        print("Usage: blender -b --python script.py -- /path/to/file.blend")
        sys.exit(1)
    
    return argv[0]

def set_scene_camera(camera_name="robo_cam"):
    scene = bpy.context.scene
    cam = bpy.data.objects.get(camera_name)
    if cam is None or cam.type != 'CAMERA':
        print(f"ERROR: Camera '{camera_name}' not found or is not a camera.")
        sys.exit(1)
    scene.camera = cam
    print(f"Using camera: {camera_name}")

def set_render_engine(engine_name="BLENDER_EEVEE_NEXT"):
    bpy.context.scene.render.engine = engine_name
    print(f"Render engine set to: {engine_name}")

def render_offscreen(scene):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=False)

    img = bpy.data.images['Render Result']
    pixels = np.array(img.pixels[:])
    width, height = img.size
    image = (pixels * 255).astype(np.uint8).reshape((height, width, 4))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image

def stream_headless(start=1, end=50, fps=24):
    scene = bpy.context.scene
    width = scene.render.resolution_x
    height = scene.render.resolution_y

    print("Starting headless frame stream... (Press Q in window to quit early)")

    for frame in range(start, end + 1):
        scene.frame_set(frame)
        frame_img = render_offscreen(scene)

        cv2.imshow("Blender Render Stream", frame_img)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            print("Stream interrupted by user.")
            break

    cv2.destroyAllWindows()
    print("Stream finished.")

def main():
    blend_path = parse_args()

    if not os.path.exists(blend_path):
        print(f"ERROR: Blend file not found: {blend_path}")
        sys.exit(1)

    print(f"Loading .blend file: {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=blend_path)

    set_render_engine()  # or "BLENDER_EEVEE_NEXT" if supported
    set_scene_camera("robo_cam")

    stream_headless(start=1, end=100, fps=24)

main()
