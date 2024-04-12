import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, Conv_QNet
from helper import plot
import math

MAX_MEMORY = 1000
BATCH_SIZE = 50
LR = 0.001
W = 12
H = 12
P = 4
C = 3

Type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 1 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # self.model = Linear_QNet(11, 256, 3)
        self.model = Conv_QNet(C, W * P, H * P, 4).type(Type)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, Type=Type)

        # self.state = deque(maxlen=C)
        # for i in range(C):
        #     self.state.append(np.zeros((W * P, H * P)))

    # def NewState(self):
    #     self.state.clear()
    #     for i in range(C):
    #         self.state.append(np.zeros((W * P, H * P)))

    # need pre-modeling and neglect the body select(aborted)
    def get_state_s(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
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

        return np.array(state, dtype=int)

    # return an image as a state
    def get_state(self, game):
        return game.get_img(P)
        self.state.append(game.get_img(P))
        imgs = [self.state[i] for i in range(C)]
        return np.array(imgs)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games
        theta = 0.002
        self.epsilon = 2 * (1 - 1. / (1 + math.exp(- theta * self.n_games))) 

        if random.random() < self.epsilon:
            move = random.randint(0, 3)
        else:
            state0 = torch.tensor(state).type(Type)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        return move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_expectation = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(W=W, H=H)
    while agent.n_games < 1000:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        # agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        agent.train_long_memory()

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            # agent.NewState()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score)

            expectation = agent.model(torch.tensor(agent.get_state(game)).type(Type))
            plot_expectation.append(np.max(expectation.cpu().detach().numpy()))
            # agent.state.pop()

            print('Expectation', plot_expectation[-1])

            plot_scores.append(score)

            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores, plot_expectation)
    
    print('Record ' + str(record) + ' Mean ' + str(total_score / agent.n_games))


if __name__ == '__main__':
    train()