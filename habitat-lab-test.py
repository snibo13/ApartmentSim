import gym
import habitat.gym

# Load embodied AI task (RearrangePick) and a pre-specified virtual robot
env = gym.make("HabitatRenderPick-v0")
observations = env.reset()

terminal = False

# Step through environment with random actions
while not terminal:
    observations, reward, terminal, info = env.step(env.action_space.sample())
