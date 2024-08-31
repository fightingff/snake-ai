import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QTrainer, Conv_QNet
from helper import plot
import math
from time import sleep
import sys
import copy

MAX_MEMORY = 10000
BATCH_SIZE = 128
LR = 1e-5
W = 32
H = 32
P = 1
C = 1

Type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
class Agent:

    def __init__(self):
        self.n_games = 0
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Conv_QNet(C, W * P, H * P, 4).type(Type)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, Type=Type)

        self.state = deque(maxlen=C)

    def get_state_s(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            game.is_collision(point_l),  # danger left
            game.is_collision(point_r),  # danger right
            game.is_collision(point_u),  # danger up
            game.is_collision(point_d),  # danger down
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int).reshape(1, -1)

    # return an image as a state as well as a status vector
    def get_state(self, game):
        
        # get the status vector: [food_x, food_y, head_x, head_y, direction]
        vec = self.get_state_s(game)
        # print(vec)
        return (game.get_img(P), np.array(vec).reshape(1, -1))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state[0], state[1], action, reward, next_state[0], next_state[1], done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self, temp_model):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states_img, states_vec, actions, rewards, next_states_img, next_states_vec, dones = zip(*mini_sample)
        self.trainer.train_step(states_img, states_vec, actions, rewards, next_states_img, next_states_vec, dones, temp_model)

    def get_action(self, state, game, epsilon, greedy):
        # random moves: tradeoff exploration / exploitation
        # theta = 1e-5
        # self.epsilon = 2 * (1 - 1. / (1 + math.exp(- theta * self.n_games))) 

        if random.random() < epsilon:
            # use two strategies
            # 1. random move
            # 2. greedy move towards the food with decreasing probability
            if random.random() < greedy:
                # greedy move
                # 0 - right 1 - left 2 - up 3 - down
                head = game.snake[0]
                food = game.food
                if food.x == head.x:
                    if food.y < head.y:
                        move = 3
                    else:
                        move = 1
                elif food.y == head.y:
                    if food.x < head.x:
                        move = 2
                    else:
                        move = 0
                else:
                    if food.y < head.y:
                        move = 3
                    else:
                        move = 1
            else:
                move = random.randint(0, 3)

        else:
            self.model.eval()
            prediction = self.model(state[0], state[1])
            move = torch.argmax(prediction).item()
            self.model.train()

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_expectation = []
    total_score = 0
    record = 0
    agent = Agent()
    agent.model.load()
    game = SnakeGameAI(W=W, H=H)
    while agent.n_games < 1000:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old, game, epsilon=0.9 - agent.n_games / 30000, greedy=0.99)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            for epoch in range(10):
                temp_model = copy.deepcopy(agent.model)
                agent.train_long_memory(temp_model)
            agent.model.save()

            game.reset()
            agent.n_games += 1
            print('Game', agent.n_games, 'Score', score)
            plot_scores.append(score)

            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)
    
def test():
    agent = Agent()
    agent.model.load()
    agent.model.eval()
    game = SnakeGameAI(W=W, H=H)
    while True:
        state_old_img, state_old_vec = agent.get_state(game)
        state_old_img = torch.Tensor(state_old_img).type(Type)
        state_old_vec = torch.Tensor(state_old_vec).type(Type)
        moves = agent.model(state_old_img, state_old_vec).cpu().detach().numpy().flatten()

        print(moves)
        final_move = np.argmax(moves)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        sleep(0.1)
        if done:
            print('Score', score)
            break

def greedy():
    game = SnakeGameAI(W=W, H=H)
    while True:
        head = game.snake[0]
        food = game.food
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        count_l, count_r, count_u, count_d = 0, 0, 0, 0
        for pt in game.snake[1:]:
            if pt.x < head.x:
                count_l += 1
            elif pt.x > head.x:
                count_r += 1
            elif pt.y < head.y:
                count_u += 1
            else:
                count_d += 1

        if food.x < head.x and not game.is_collision(point_l) and count_l == 0: # food left
            move = 2
        elif food.x > head.x and not game.is_collision(point_r) and count_r == 0: # food right
            move = 0
        elif food.y < head.y and not game.is_collision(point_u) and count_u == 0: # food up
            move = 3
        elif food.y > head.y and not game.is_collision(point_d) and count_d == 0: # food down
            move = 1
        else:
            directions = []
            if not game.is_collision(point_l):
                directions.append((2, count_l))
            if not game.is_collision(point_r):
                directions.append((0, count_r))
            if not game.is_collision(point_u):
                directions.append((3, count_u))
            if not game.is_collision(point_d):
                directions.append((1, count_d))
            
            if len(directions) != 0:
                move = min(directions, key=lambda x: x[1])[0]
            else:
                move = random.randint(0, 3)
        
        print(move)
        reward, done, score = game.play_step(move)
        sleep(0.1)
        if done:
            print('Score', score)
            break

def cover():
    game = SnakeGameAI(W=W, H=H)
    # make a covering path of the whole grid

    # 1. go right
    while not game.is_collision(Point(game.snake[0].x + 1, game.snake[0].y)):
        game.play_step(0)
    
    # 2. keep go the full column
    col = H - 1
    while True:
    
        if col % 2 == 1:
            while not game.is_collision(Point(game.snake[0].x, game.snake[0].y + 1)):
                reward, done, score = game.play_step(1)
        else:
            while not game.is_collision(Point(game.snake[0].x, game.snake[0].y - 1)) and game.snake[0].y - 1 > 0:
                reward, done, score = game.play_step(3)
        
        if col == 0:
            reward, done, score = game.play_step(3)
            # go right
            while not game.is_collision(Point(game.snake[0].x + 1, game.snake[0].y)):
                reward, done, score = game.play_step(0)
            col = H - 1
        else:
            reward, done, score = game.play_step(2)
            col -= 1


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        train()
    else:
        if(args[1] == 't'):
            test()
        elif(args[1] == 'g'):
            greedy()
        else:
            cover()