import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Conv_QNet(nn.Module):
    def __init__(self, input_c, input_w, input_h, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(input_c),
            nn.Conv2d(input_c, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=2, stride=2),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=2, stride=2),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(input_w // 4 * input_h // 4 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        return self.model(x)
    
    def save(self, file_name='cmodel.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma, Type):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.Type = Type

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state).type(self.Type)
        next_state = torch.tensor(next_state).type(self.Type)
        action = torch.tensor(action).type(self.Type)
        reward = torch.tensor(reward).type(self.Type)
        # (n, x)
        # print(state.shape)
        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        self.optimizer.zero_grad()
        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



