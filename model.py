import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import copy
import shutil
from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)

def InNet(img, device):
    img = np.array(img)
    img = torch.Tensor(img).to(device)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    return img

class Conv_QNet(nn.Module):
    def __init__(self, input_c, input_w, input_h, output_size):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iter = 0
        # self.model = nn.Sequential(
        #     nn.Conv2d(input_c, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(128 * (input_w // 2) * (input_h // 2), 128),
        #     nn.ReLU(),
        #     nn.Linear(128, output_size)
        # ).to(self.device)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_c * input_w * input_h, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        ).to(self.device)


    def forward(self, img):
        img = np.array(img)
        img = torch.Tensor(img).to(self.device)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        return self.model(img)
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self, path='model/model.pth'):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print('Model loaded')

class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.HuberLoss()
        self.device = device

    def train_step(self, state, action, reward, next_state, done, temp_model, alpha=0.1):
        writer = SummaryWriter('runs')

        # 1: predicted Q values with next state
        future = temp_model(next_state)
        
        self.optimizer.zero_grad()
        # 2: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()

        action = torch.Tensor(action).to(self.device)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new += self.gamma * torch.max(future[idx])

            Q_old = target[idx][torch.argmax(action[idx]).item()]
            target[idx][torch.argmax(action[idx]).item()] = (1 - alpha) * Q_old + alpha * Q_new
    
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        # tensorboard
        writer.add_scalar('Loss/train', loss, self.model.iter)
        self.model.iter += 1
        writer.close()


# Actor-Critic + CNN
class ActorCritic(nn.Module):
    def __init__(self, input_c, input_w, input_h, output_size):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iter = 0
        feature_size = 256
        self.actor = nn.Sequential(
            nn.Conv2d(input_c, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(feature_size, feature_size // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear((feature_size // 2) * (input_w // 2) * (input_h // 2), feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, output_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Conv2d(input_c, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(feature_size, feature_size // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear((feature_size // 2) * (input_w // 2) * (input_h // 2), feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 1)
        ).to(self.device)

    def forward(self, img):
        img = InNet(img, self.device)
        return self.actor(img)
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name + '.tmp')
        torch.save(self.state_dict(), file_name)
        # use copy to protect the original model
        shutil.copy(file_name, file_name[:-4])
    
    def load(self, path='model/model.pth'):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print('Model loaded')

class ACTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer_actor = optim.Adam(model.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(model.critic.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.device = device

    def train_step(self, state, action, reward, next_state, done, alpha=0.1):
        writer = SummaryWriter('runs')

        state = InNet(state, self.device)
        next_state = InNet(next_state, self.device)
        reward = torch.Tensor(reward).view(-1, 1).to(self.device)
        done = torch.Tensor(done).view(-1, 1).to(self.device)
        action = torch.Tensor(action).view(-1, 1).to(self.device)


        target = self.model.critic(next_state) * self.gamma * (1 - done) + reward
        delta = target - self.model.critic(state)
        # print(self.model.actor(state))
        prob = self.model.actor(state).gather(1, action.long())
        prob += 1e-5
        prob = torch.log(prob)
        # print(prob)
        # if torch.isnan(prob).any():
        #     print(delta)
        #     print(prob)
        #     print(self.model.actor(state))
        #     exit()

        
        # avoid nan
        loss_actor = torch.mean(-prob * delta.detach())
        # loss_actor = torch.clamp(loss_actor, -1, 1)
        loss_critic = torch.mean(self.criterion(self.model.critic(state), target))

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        loss_actor.backward()
        loss_critic.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        # tensorboard
        writer.add_scalar('Loss/train_actor', loss_actor, self.model.iter)
        writer.add_scalar('Loss/train_critic', loss_critic, self.model.iter)
        self.model.iter += 1
        writer.close()

# PPO-TRAINER
class PPOTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.lamda = 0.99
        self.EPOCH = 10
        self.EPS = 0.2
        self.model = model
        self.optimizer_actor = optim.Adam(model.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(model.critic.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.device = device

    def train_step(self, state, action, reward, next_state, done, alpha=0.1):
        writer = SummaryWriter('runs')

        state = InNet(state, self.device)
        next_state = InNet(next_state, self.device)
        reward = torch.Tensor(reward).view(-1, 1).to(self.device)
        done = torch.Tensor(done).view(-1, 1).to(self.device)
        action = torch.Tensor(action).view(-1, 1).to(self.device)

        with torch.no_grad():
            target = self.model.critic(next_state) * self.gamma * (1 - done) + reward
            delta = target - self.model.critic(state)

            # print(self.model.actor(state))
            prob_old = self.model.actor(state).gather(1, action.long())
            prob_old = torch.log(prob_old + 1e-5)

            # compute advantage GAE
            advantage = torch.zeros_like(reward)
            adv = 0
            for i in reversed(range(len(reward))):
                adv = delta[i] + self.gamma * self.lamda * adv
                advantage[i] = adv
            advantage = advantage.to(self.device)

        for i in range(self.EPOCH):
            prob_new = self.model.actor(state).gather(1, action.long())
            prob_new = torch.log(prob_new + 1e-5)

            r = torch.exp(prob_new - prob_old)
            surr1 = r * advantage
            surr2 = torch.clamp(r, 1 - self.EPS, 1 + self.EPS) * advantage

            loss_actor = torch.mean(-torch.min(surr1, surr2))
            loss_critic = torch.mean(self.criterion(self.model.critic(state), target))
            loss = loss_actor + loss_critic
        

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            # loss_actor.backward(retain_graph=True)
            # loss_critic.backward(retain_graph=True)
            
            loss.backward(retain_graph=True)
            self.optimizer_actor.step()
            self.optimizer_critic.step()

            # tensorboard
            writer.add_scalar('Loss/train_actor', -loss_actor, self.model.iter)
            writer.add_scalar('Loss/train_critic', loss_critic, self.model.iter)
            self.model.iter += 1
        
        writer.close()