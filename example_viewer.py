import habitat_sim
import habitat_sim.utils.common as utils
import numpy as np
import cv2
import os

def make_cfg(scene_dataset_config_file: str, scene_name: str):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config_file
    sim_cfg.scene_id = scene_name
    sim_cfg.enable_physics = False

    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.uuid = "color_sensor"
    sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sensor_cfg.resolution = [512, 512]
    sensor_cfg.position = [0.0, 1.5, 0.0]
    sensor_cfg.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

# Paths
scene_config_file = "custom.scene_dataset_config.json"
scene_name = "my_apartment"

# Load simulator
cfg = make_cfg(scene_config_file, scene_name)
sim = habitat_sim.Simulator(cfg)

# Set agent initial state
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([1.0, 1.6, 1.0])  # Small offset forward
# agent_state.rotation = quat_from_angle_axis(np.pi / 2, np.array([0, 1, 0]))
sim.initialize_agent(0, agent_state)

# Get observation
obs = sim.get_sensor_observations()
rgb_img = obs["color_sensor"]

# Show with OpenCV
cv2.imshow("Custom Habitat Scene", rgb_img[..., ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
