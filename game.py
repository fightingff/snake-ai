import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
YELLOW1 = (255, 100, 0)
YELLOW2 = (255, 255, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 100
class SnakeGameAI:

    def __init__(self, W=32, H=32):
        self.w = W * BLOCK_SIZE
        self.h = H * BLOCK_SIZE
        self.W = W
        self.H = H
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.W//2, self.H//2)
        self.snake = [self.head,
                      Point(self.head.x-1, self.head.y),
                      Point(self.head.x-2, self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(1, self.W - 2)
        y = random.randint(1, self.H - 2)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        dis = np.linalg.norm(np.array([self.head.x, self.head.y]) - np.array([self.food.x, self.food.y]))
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        reward = dis - np.linalg.norm(np.array([self.head.x, self.head.y]) - np.array([self.food.x, self.food.y]))
        reward = 1. / len(self.snake) if reward > 0 else - 1. / len(self.snake)

        # 3. check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -1
            self.snake.pop()
            # return reward, game_over, self.score

        # 4. place new food or just move
        elif self.head == self.food:
            self.score += 1
            reward = 2
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x >= self.W - 1 or pt.x <= 0 or pt.y >= self.H - 1 or pt.y <= 0:
            return True
        # hits itself
        # if pt in self.snake[1:]:
        #     return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)
        for i in range(self.H):
            pygame.draw.rect(self.display, GREEN, pygame.Rect(0, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN, pygame.Rect((self.W - 1) * BLOCK_SIZE, i * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        for i in range(self.W):
            pygame.draw.rect(self.display, GREEN, pygame.Rect(i * BLOCK_SIZE, 0, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN, pygame.Rect(i * BLOCK_SIZE, (self.H - 1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.snake:
            if(pt != self.head):
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, YELLOW1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, YELLOW2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [Right, Left, Up, Down]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        self.direction = clock_wise[action]

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)

    def get_img(self, P=1):
        # draw the map
        S = self.W * self.H
        img = np.ones((3, self.W * P, self.H * P))
        for i in range(self.H * P):
            for j in range(P):
                img[[0,2], j, i] = img[[0,2], self.W * P - P + j, i] = 0
        
        for i in range(self.W * P):
            for j in range(P):
                img[[0,2], i, j] = img[[0,2], i, self.H * P - P + j] = 0
        
        # draw snake
        for _, pt in enumerate(self.snake):
            for i in range(P):
                for j in range(P):
                    img[[0, 1], pt.x * P + i, pt.y * P + j] = _ * 0.1
        
        # draw head
        for i in range(P):
            for j in range(P):
                img[2, self.head.x * P +i, self.head.y * P + j] = 0

        # draw food
        for i in range(P):
            for j in range(P):
                img[[1,2], self.food.x * P + i, self.food.y * P + j] = 0
        
        # plt.imshow(img.transpose(1, 2, 0))
        # input()
        return img
