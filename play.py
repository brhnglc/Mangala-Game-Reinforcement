from MangalaGym import MangalaEnv
import pygame
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

env = MangalaEnv()
env.render_mode = "human"
    
input_size = len(env.get_observation())
output_size = env.action_space.n

loaded_model = DQN(input_size, output_size)
loaded_model.load_state_dict(torch.load('models//130000-trained_model-13-09_42.pth')) 
loaded_model.eval() 


state = torch.tensor([env.reset()], dtype=torch.float32)
total_reward = 0
while True:    
    if(env.player_turn == 1):
        raw_act = loaded_model(state) 
        enum_raw_act = list(enumerate(raw_act[0].detach().numpy()))
        sorted_enum_raw_act = list(reversed(sorted(enum_raw_act, key=lambda x: x[1])))
        action2 = "none"
        state_np = state.detach().numpy()[0]
        enum_raw_act = list(enumerate(raw_act[0].detach().numpy()))
        sorted_enum_raw_act = list(reversed(sorted(enum_raw_act, key=lambda x: x[1])))
        
        action = "none"
        state_np = state.detach().numpy()[0]
        state_pos = 0
        for i,_ in sorted_enum_raw_act:
            state_pos = i
            if(state_np[state_pos] != 0):
                action = i
                break        
        time.sleep(1)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = torch.tensor([next_state], dtype=torch.float32)
        print("observation->",state,"action->",action,"reward->",reward)

    else:
        running = True
        quitt = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.close()
                elif event.type == pygame.KEYDOWN and event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_6:
                    move = event.key - pygame.K_1
                    running = False
        next_state, reward, done, _ = env.step(move)
        state = torch.tensor([next_state], dtype=torch.float32)
        print("Observation->",next_state,"action->",move,"reward->",reward) 
    print("*"*120)
    if done:
        break

print(f"Total Test Reward: {total_reward}")


