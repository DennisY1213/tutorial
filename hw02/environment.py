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

class Environment:
    def __init__(self, track):
        self.track = track
        rows, cols = list(np.where(track == START))
        self.start_set = list(zip(rows, cols))
        rows, cols = list(np.where(track == GRAVEL))
        self.gravel_set = list(zip(rows, cols))
#         self.state = self.start_set[np.random.choice([i for i in range(len(start_set))])]

        self.num_action = 9

        self.screen = None
        self.clock = None

    ##Return a start position
    def reset(self):
        state = self.start_set[np.random.choice([i for i in range(len(self.start_set))])]
        return state, [0,0]

    def check_finish(self, state):
        rows, cols = np.where(self.track == FINISH)
        x, y = state
        if x in rows and y >= cols[0]:
            return True
        return False

    def check_crash(self, old_state, new_state):
        x_new, y_new = new_state
        x_old, y_old = old_state
        ##check if the new state is out of track
        if x_new < 0 or x_new >= GRID_SIZE or y_new < 0 or y_new >= GRID_SIZE or self.track[x_new][y_new] == GRAVEL:
            return True

        ##check if the projected path intersects the boundary
        for r in range(x_old, x_new + 1):
            if self.track[r, y_old] == GRAVEL:
                return True
        for c in range(y_old, y_new + 1):
            if self.track[x_old, c] == GRAVEL:
                return True

        return False

    def take_action(self, state, speed, action):
        ##speeds on both directions >= 0 and < 5
        ##speeds on both directions cannot be 0 at the same time
        ##speeds has 10% chance of not increase
        reward = -1
        new_x_speed, new_y_speed = 0, 0
        x_acc, y_acc = action

        threshold = np.random.rand()
        if threshold <= 0.1:
            ##speeds don't change
            new_x_speed, new_y_speed = speed[0], speed[1]
        else:
            new_x_speed = speed[0] + x_acc
            if new_x_speed < 0: new_x_speed = 0
            if new_x_speed > 4: new_x_speed = 4

            ## if the x speed is already 0, then y speed cannot be 0
            if new_x_speed == 0 and speed[1] + y_acc == 0:
                new_y_speed = speed[1] ##doesn't change
            else:
                new_y_speed = speed[1] + y_acc
                if new_y_speed < 0: new_y_speed = 0
                if new_y_speed > 4: new_y_speed = 4

        new_state = (state[0] - new_x_speed, state[1] + new_y_speed)
        new_speed = [new_x_speed, new_y_speed]

        terminated = False
        if self.check_finish(new_state):
            terminated = True
            rows, cols = np.where(self.track == FINISH)
            new_state = (new_state[0], cols[0])
            # new_speed = [0, 0]
        elif self.check_crash(state, new_state):
            new_state = self.start_set[np.random.choice([i for i in range(len(self.start_set))])]
            new_speed = [0, 0]


        return reward, terminated, new_state, new_speed


    def draw_grid(self, grid, state):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if grid[x][y] == START:
                    color = START_COLOR
                    pygame.draw.rect(self.screen, START_COLOR, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)

                elif grid[x][y] == FINISH:
                    color = FIN_COLOR
                    pygame.draw.rect(self.screen, FIN_COLOR, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)

                if grid[x][y] == TRACK:
                    color = TRACK_COLOR
                elif grid[x][y] == GRAVEL:
                    color = GRAVEL_COLOR
                pygame.draw.rect(self.screen, color, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        ##Draw the car
        pygame.draw.rect(self.screen, CAR_COLOR, (state[1] * CELL_SIZE, state[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)


    def display(self, state):
        if self.screen == None:
          # Initialize Pygame
          pygame.init()
          # Create a Pygame window
          self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
          pygame.display.set_caption("Race Track")
          self.clock = pygame.time.Clock()

        self.screen.fill(WHITE)
        self.draw_grid(self.track, state)

        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window = None
                pygame.quit()
                self.truncated = True
        self.clock.tick(FPS)


