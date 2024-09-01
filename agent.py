import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QTrainer, Conv_QNet, ActorCritic, ACTrainer, PPOTrainer
from helper import plot
import matplotlib.pyplot as plt
import math
from time import sleep
import sys
import copy
from tqdm import tqdm
import tensorboardX

MODEL = 'model/model_ppo.pth'

# Memory
MAX_MEMORY = 10000

# Exploration settings
N_GAMES = 10000
EPS_START = 0
EPS_END = 0

BATCH_SIZE = 128 
EPOCH = 10

# Game settings
LR = 1e-5
W = 8
H = 8
P = 1
C = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Agent:

    def __init__(self):
        self.n_games = 0
        self.gamma = 1 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # simple DQN + CNN --> weak
        # self.model = Conv_QNet(C, W * P, H * P, 4)
        # self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=device)

        # simple Actor-Critic + CNN --> about score 10
        # self.model = ActorCritic(C, W * P, H * P, 4)
        # self.trainer = ACTrainer(self.model, lr=LR, gamma=self.gamma, device=device)

        # PPO + CNN --> about score 15
        self.model = ActorCritic(C, W * P, H * P, 4)
        self.trainer = PPOTrainer(self.model, lr=LR, gamma=self.gamma, device=device)

    # return an image as a state
    def get_state(self, game):
        return game.get_img(P)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self, temp_model=None):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        if len(mini_sample) == 0:
            return
        
        state, action, reward, next_state, done = zip(*mini_sample)
        if temp_model is not None:
            self.trainer.train_step(state, action, reward, next_state, done, temp_model)
        else:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory_ppo(self):
        if len(self.memory) <= 1:
            return
        sample = self.memory
        state, action, reward, next_state, done = zip(*sample)
        self.trainer.train_step(state, action, reward, next_state, done)
        self.memory.clear()

    def get_action(self, state, game, epsilon, greedy):
        # random moves: tradeoff exploration / exploitation

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
            with torch.no_grad():
                prediction = self.model(state)
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
    agent.model.load(path=MODEL)
    game = SnakeGameAI(W=W, H=H)
    R = 0
    S = 0
    writer = tensorboardX.SummaryWriter('reward')

    while agent.n_games < N_GAMES:
        # get old state
        state_old = agent.get_state(game)

        # get move
        EPS = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * agent.n_games / N_GAMES)
        final_move = agent.get_action(state_old, game, epsilon=EPS, greedy=0.8)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        R += reward

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # if len(game.snake) >= 10:
        #     agent.remember(state_old, final_move, reward, state_new, done)
        # else:
        #     if random.random() < 0.5:
        #         agent.remember(state_old, final_move, reward, state_new, done)

        # if reward > 0:
        #     agent.remember(state_old, final_move, reward, state_new, done)
        # else:
        #     if random.random() < 0.5:
        #         agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            # for epoch in range(EPOCH):
                # agent.train_long_memory()
            agent.train_long_memory_ppo()
            # if game.score >= 10:
            #     agent.train_long_memory_ppo()
            # else:
            #     agent.memory.clear()
            agent.model.save(file_name=MODEL.split('/')[-1])

            # game reset
            S += score
            game.reset()
            agent.n_games += 1
            if agent.n_games % 100 == 0:
                print('Game', agent.n_games, 'Score', S / 100.0)
                S = 0
            
            writer.add_scalar('reward', R, agent.n_games)
            R = 0

            # plot
            # print('Game', agent.n_games, 'Score', score)

            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_scores.append(mean_score)

            # plot(plot_scores, plot_mean_scores)
            writer.close()
    
def test():
    agent = Agent()
    agent.model.load(path=MODEL)
    agent.model.eval()
    game = SnakeGameAI(W=W, H=H)
    while True:
        state_old = agent.get_state(game)
        moves = agent.model(state_old).cpu().detach().numpy().flatten()

        final_move = np.argmax(moves)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        print(moves, np.argmax(moves), reward)
        sleep(0.1)
        if done:
            print('Score', score)
            break

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        train()
    else:
        test()