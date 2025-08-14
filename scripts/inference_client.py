from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
import time
from opencv import OpenCVCamera, OpenCVCameraConfig
import franky 

robot = franky.Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.02
robot.set_collision_behavior([15.0]*7 , [30.0]*6)
gripper = franky.Gripper("172.16.0.2")
max_gripper_width = 0.08  # ~80 mm
FPS = 30

# Camera setup (reuse your config)
camera_configs = {
    "left": OpenCVCameraConfig(camera_index=10, fps=FPS, width=640, height=480),
    "right": OpenCVCameraConfig(camera_index=16, fps=FPS, width=640, height=480),
    "wrist": OpenCVCameraConfig(camera_index=4, fps=FPS, width=640, height=480),
}

cameras = {}
for name, cfg in camera_configs.items():
    cameras[name] = OpenCVCamera(cfg)
    try:
        cameras[name].connect()
        cameras[name].async_read()
        print(f"Connected to camera {name} (index {cfg.camera_index})")
    except Exception as e:
        print(f"Failed to connect camera {name}: {e}")

# Initialize policy client
client = websocket_client_policy.WebsocketClientPolicy(host="192.168.3.20", port=8000)

num_steps = 100  # Set your desired number of steps
task_instruction = "Pick the bowl and place it in the green square"

for step in range(num_steps):
    try:
        # Read images
        left_img = cameras["left"].async_read()
        right_img = cameras["right"].async_read()
        wrist_img = cameras["wrist"].async_read()

        # Skip bad frames
        if any(
            (np.mean(img, axis=(0, 1)) == 0.0).any() or
            (np.mean(img, axis=(0, 1)) <= 50.0).any()
            for img in (wrist_img, left_img, right_img)
        ):
            print("Invalid image, skipping step.")
            continue

        # Get robot state
        state = robot.current_joint_state.position.tolist()
        gripper_width = [gripper.width/2.0]
        state = np.array(state + gripper_width)
        state_corrected = np.pad(state, (0, 32 - state.shape[0]), mode="constant")
        # Match training DataDict structure
        observation = {
            "images": {
                "base_0_rgb": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(left_img, 224, 224)
                ),
                "right_wrist_0_rgb": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(right_img, 224, 224)
                ),
                "left_wrist_0_rgb": image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, 224, 224)
                ),
            },
            "state": state_corrected, 
            "prompt": "0",
        }

        # Inference
        result = client.infer(observation)
        action_chunk = result["actions"]
        print(action_chunk[:, :7].shape)
        
        robot.move(franky.JointMotion(action_chunk[0, :7].tolist()))

        print(f"Step {step}: Executed action.")

    except Exception as e:
        print(f"Step {step}: Error - {e}")
        continue


    time.sleep(0.1)  # Adjust sleep for your control rate