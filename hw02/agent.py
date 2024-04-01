import pygame
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
import os
sys.path.append('~/Desktop/Tufts-CS/CS138')
from track import build_track
# Constants
GRID_SIZE = 32
CELL_SIZE = 20
FPS = 60
GRID_WIDTH = GRID_SIZE * CELL_SIZE
GRID_HEIGHT = GRID_SIZE * CELL_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
TRACK_COLOR = (160, 160, 160)
GRAVEL_COLOR = (255, 255, 255)
FIN_COLOR = (255, 0, 0)
START_COLOR = (0, 255, 0)
CAR_COLOR = (0, 0, 255)
GRAVEL = -1
TRACK = 0
START = 1
FINISH = 2
num_action = 9
# ##build the track
track = build_track()

class Agent:
    def __init__(self, epsilon = 0.1, gamma = 0.9):
        self.epsilon = epsilon
        self.gamma = gamma
        self.speed = [0, 0]   ##[horizontal speed, vertical speed]

        self.actions = {
            0: [1, 1],
            1: [0, -1],
            2: [0, 1],
            3: [1, 0],
            4: [1, -1],
            5: [0, 0],
            6: [-1, 0],
            7: [-1, -1],
            8: [-1, 1]
        }

        self.state = [0, 0]
        self.num_actions = len(self.actions)
        self.Q = np.zeros((GRID_SIZE, GRID_SIZE, self.num_actions))   # Initialize action 
        self.N = np.zeros((GRID_SIZE, GRID_SIZE, self.num_actions))  # Visit count for each state-action pair
        self.C = np.zeros((GRID_SIZE, GRID_SIZE, self.num_actions))
        self.target_policy = np.argmax(self.Q, axis = -1)

    def reset(self, start_state):
        self.state = start_state
        self.speed = [0, 0]

    ##Q is the q value of a given state, should be an array of size 9
    def soft_policy(self, Q):
        epsilon = self.epsilon
        num_actions = self.num_actions
        policy = np.ones(num_actions) * epsilon / num_actions
        A_opt = np.argmax(Q)
        policy[A_opt] = 1 - epsilon + epsilon / num_actions
        # print(f"policy is: {policy}")
        return policy

    def mc_control(self, env, num_episodes, on_policy = True):
        print("here")
        for i,j in env.start_set:
            self.Q[i][j][5] = -1000
            self.Q[i][j][7] = -1000
            self.target_policy = np.argmax(self.Q, axis = -1)
        episode_len = []
        for episode in range(num_episodes):
            print(f"episode {i} starts")
            self.state, self.speed = env.reset()
            episode_states = []
            policy = None

            # Generate an episode using ε-soft policy
            while True:
                policy = self.soft_policy(self.Q[self.state])
                action = np.random.choice(np.arange(self.num_actions), p=policy)
                if self.state in env.start_set and action in [1,5,7]:
                    action = 2
                reward, terminated, new_state, new_speed = env.take_action(self.state, self.speed, self.actions[action])
                # uncomment the following line to watch the training process
                # env.display(self.state) 
                
                episode_states.append((self.state, action, reward))
                self.state = new_state
                self.speed = new_speed


                if terminated:
                    break
            episode_len.append(len(episode_states))
            G = 0.0
            if on_policy:
                ##on-policy MC control
                for t in reversed(range(len(episode_states))):
                    state, action, reward = episode_states[t]
                    i, j = state
                    G = self.gamma * G + reward

                    self.N[i][j][action] += 1
                    self.Q[i][j][action] += 1 / self.N[i][j][action] * (G - self.Q[i][j][action])
            
            else:
                W = 1.0
                for t in reversed(range(len(episode_states))):
                    state, action, reward = episode_states[t]
                    i, j = state
                    G = self.gamma * G + reward

                    self.C[i][j][action] += W
                    self.Q[i][j][action] += W / self.C[i][j][action] * (G - self.Q[i][j][action])
                    self.target_policy[i][j] = np.argmax(self.Q[i][j])
#                     print(self.target_policy[i][j], action)
                    if self.target_policy[i][j] != action:
                        break
                    W = W * 1 / policy[action]
                
        return episode_len
    