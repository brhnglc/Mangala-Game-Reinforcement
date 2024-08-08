from MangalaGym import MangalaEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

from collections import namedtuple, deque
import random
from datetime import datetime
from itertools import count

#matplotlip setup
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cpu")



BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
num_episodes = 10000001
PER_RND_DEPLOY = 500
PER_SAVE_MODEL = 10000
opponet_net_dif = 10000

opennig_move = {"0-0":0,"0-1":0,"0-2":0,"0-3":0,"0-4":0,"0-5":0,"1-0":0,"1-1":0,"1-2":0,"1-3":0,"1-4":0,"1-5":0,"2-0":0,"2-1":0,"2-2":0,"2-3":0,"2-4":0,"2-5":0,"3-0":0,"3-1":0,"3-2":0,"3-3":0,"3-4":0,"3-5":0,"4-0":0,"4-1":0,"4-2":0,"4-3":0,"4-4":0,"4-5":0,"5-0":0,"5-1":0,"5-2":0,"5-3":0,"5-4":0,"5-5":0}
winone = 0
wintwo = 0
draw = 0


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    

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
    

def save_model(model,episode):
    simdi = datetime.now()
    gun = simdi.day
    saat = simdi.hour
    dakika = simdi.minute
    current_time = f"{gun}-{saat:02d}_{dakika:02d}"
    torch.save(model, f'{episode}-trained_model-{current_time}.pth')
    print(f'saved {episode}-trained_model-{current_time}')
    

#**********************
env = MangalaEnv()
env.render_mode = "train"

#**********************



n_actions = env.action_space.n
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device) 
target_net.load_state_dict(policy_net.state_dict())

opponet_net = DQN(n_observations, n_actions).to(device)
opponet_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state,player):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            if(player == "1"):            
                raw_act = policy_net(state)
            else:
                raw_act = opponet_net(state)
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
            return torch.tensor([[action]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[np.random.choice(np.nonzero(state[:6])[0])]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001) 
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
            
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()# updata weights
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)# gradyan kırpma smoothing
    optimizer.step()
    
    
    
total_reward = 0
capture_p1 = 0
capture_p2 = 0
go_again = 0
for i_episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    if(i_episode%opponet_net_dif == 0):
        print("Opponet Upraged Model - ",i_episode)
        opponet_net.load_state_dict(policy_net.state_dict())
        first_move = 0
    for t in count():
        if(env.player_turn == 1):

            action = select_action(state,"1")
            observation, reward, done, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
         
            if(t == 0):
                first_move = action.item()
            elif(t==1):
                opennig_move[f"{first_move}-{action.item()}"] += 1

                
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
  
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)
            
        else:
            #print("PlayerTurn2","*"*50)
            action = select_action(torch.tensor([env.turn_table(env.get_observation())], dtype=torch.float32), "2")  
            observation, reward, done, _ = env.step(action.item())
            #print(t,".","observation->",state,"next_state->",observation,"action->",action,"reward->",reward)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state
            
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            if observation[13]>observation[6]:
                wintwo += 1
            elif observation[13]<observation[6]:
                winone += 1
            else:
                draw += 1            
            capture_p1 += env.capture_amount_p1
            capture_p2 += env.capture_amount_p2
            go_again += env.go_again_amount
            break
    if(i_episode%PER_RND_DEPLOY == 0 and i_episode >1):
        opennig_sorted = dict(sorted(opennig_move.items(), key=lambda x:x[1],reverse=True))
        opennig_sorted_f5 = dict(list(opennig_sorted.items())[0:5])
        print(f"ÇAĞ: {i_episode} Reward: {total_reward} Draw:{draw} Player1:{winone} Player2:{wintwo} Capture_mean:{capture_p1/PER_RND_DEPLOY} Capture_mean_enem:{capture_p2/PER_RND_DEPLOY} Go_again_mean:{go_again/PER_RND_DEPLOY} opennig_moves:{opennig_sorted_f5}")
        capture_p1 = 0
        capture_p2 = 0
        go_again = 0
    if(i_episode%PER_SAVE_MODEL == 0 and i_episode>1):
        save_model(policy_net.state_dict(),i_episode)

        
        

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
