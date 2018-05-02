import gym
import time
env = gym.make("Pong-v0")
env.unwrapped.get_action_meanings()

env.reset()
for action in range(6):
  # time.sleep(5)
  print("action {}".format(action))
  observation, reward, done, info = env.step(action)
  env.render()
  print(reward)

env.close()