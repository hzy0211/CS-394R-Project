import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import math
from utils import *
from System_Status import System_Status
from Read_Layer import Read_Layer
from baseline import always_batching, no_batching
import seaborn as sns

class Env_static:
    def __init__(self, rt_table, new_req_seq, max_job, n_layers=38):
        assert rt_table.shape[1] == n_layers
        self.n_layers = n_layers
        self.rt_table = rt_table
        self.new_req_seq = new_req_seq
        self.max_job = max_job
        self.reset()

    def reset(self):
        self.time = 0
        self.time_till_last = 0
        self.state = np.zeros(self.n_layers)
        self.load = np.zeros(self.n_layers)
        self.job_counter = 0
        self.is_done = False
        self.observation_space = self.n_layers
        self.action_space = self.n_layers
        self.load[0] = 1
        self.load[1] = 1
        self.load[2] = 1
        self.load[3] = 0
        self.load[4] = 1
        self.state[0] = 3
        self.state[1] = 1
        self.state[2] = 5
        self.state[3] = 0
        self.state[4] = 2
        return self.state, self.load

    def new_request(self):
        while (self.job_counter < NUM_NEW_REQUEST):           
            if self.new_req_seq[self.job_counter] <= self.time and self.state.sum() < self.max_job:
                self.state[0] += 1
                self.load[0] = 1
                self.job_counter += 1
            else:
                break

    def step(self, action):
        layer_select = action
        running_load = self.state[layer_select]
        n_jobwaiting = self.state.sum()
        self.state[layer_select] = 0
        self.load[layer_select] = 0
        if layer_select + 1 < self.n_layers:
            self.state[layer_select + 1] += running_load
            self.load[layer_select + 1] = 1
        running_time = self.rt_table[max(int(running_load - 1), 0), layer_select]
        self.time += running_time
        reward = -running_time * n_jobwaiting
        if running_load == 0:
            reward = -1
        if self.state.sum() == 0:
            self.is_done = True
        return self.state, self.load, reward, self.is_done

class Reinforce(nn.Module):
    def __init__(self, hl_size, n_input, action_space):
        super(Reinforce, self).__init__()
        self.actor1 = nn.Linear(n_input, hl_size)
        self.actor2 = nn.Linear(hl_size, hl_size)
        self.actor3 = nn.Linear(hl_size, action_space)

    def forward(self, state, load):
        action_probs = self.actor1(state)
        action_probs = F.relu(action_probs)
        action_probs = self.actor2(action_probs)
        action_probs = F.relu(action_probs)
        action_probs = self.actor3(action_probs)
        action_probs = F.softmax(action_probs, dim=-1) * load
        return action_probs, 0


class Actor_Critic(nn.Module):
    def __init__(self, hl_size, n_input, action_space):
        super(Actor_Critic, self).__init__()
        self.actor1 = nn.Linear(n_input, hl_size)
        self.actor2 = nn.Linear(hl_size, hl_size)
        self.actor3 = nn.Linear(hl_size, action_space)
        self.critic1 = nn.Linear(n_input, hl_size)
        self.critic2 = nn.Linear(hl_size, hl_size)
        self.critic3 = nn.Linear(hl_size, 1)

    def forward(self, state, load):
        action_probs = self.actor1(state)
        action_probs = F.relu(action_probs)
        action_probs = self.actor2(action_probs)
        action_probs = F.relu(action_probs)
        action_probs = self.actor3(action_probs)
        action_probs = F.softmax(action_probs, dim=-1)
        action_probs = action_probs * load
        v_s = self.critic1(state)
        v_s = F.relu(v_s)
        v_s = self.critic2(v_s)
        v_s = F.relu(v_s)
        v_s = self.critic3(v_s)
        return action_probs, v_s

def compute_returns(rewards, gamma=1):
    R = 0
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R
        returns.insert(0, R)
    return returns

def get_action(state, load, model):
    state = torch.tensor(state).float()
    load = torch.tensor(load).float()
    action_probs, v_s= model(state, load)
    dist = Categorical(action_probs)
    action = dist.sample()
    return action, v_s, dist

def read_layer(curr_status, data_file):
    curr_sum = 0.0
    fp = open(data_file,"r")
    for i in range(BATCH_SIZE):
        curr_layer = fp.readline().strip()
        curr_layer = curr_layer.split()
        for j in range(LAYER_SIZE):
            curr_status.batch_matrix[i][j] = round(float(curr_layer[j]),6)
            if i==0:
                curr_sum += curr_status.batch_matrix[i][j]

    fp.close()
    group_num = GROUP_NUM
    j = 0
    for k in range(BATCH_SIZE):
        curr_status.group_batch_matrix[k][j] = 0.0

    for i in range(LAYER_SIZE):
        for k in range(BATCH_SIZE):
            curr_status.group_batch_matrix[k][j] += curr_status.batch_matrix[k][i]

        if (curr_status.group_batch_matrix[0][j]>=curr_sum/(1.0*group_num)):
            curr_sum -= curr_status.group_batch_matrix[0][j]
            group_num -= 1
            if abs(curr_sum)<1e-8:
                curr_sum=0.0
            assert (group_num>=0)
            assert (curr_sum>=0)
            j += 1

def ac_gae(rt_table, new_req_seq, n_episode=1000, gamma=1):
    env = Env_static(rt_table, new_req_seq, 90, 5)
    n_input = env.observation_space
    action_space = env.action_space
    hl_size = 128
    ac = Actor_Critic(hl_size, 5, 5)
    optimizer = optim.Adam(ac.parameters())
    reward_history = np.zeros(n_episode)
    best = -1
    for i in range(n_episode):
        state, load = env.reset()
        saved_logprobs = []
        saved_values = []
        rewards = []
        for t in range(1000):
            action, v_s, dist = get_action(state, load, ac)
            log_prob = dist.log_prob(action).unsqueeze(0)
            saved_logprobs.append(log_prob)
            saved_values.append(v_s)
            state, load, reward, is_done = env.step(action.item())
            rewards.append(torch.tensor([reward], dtype=torch.float))
            reward_history[i] += reward
            if i == n_episode - 1:
                print('time: {}, state: {}'.format(env.time, env.state))
                print('load: {}'.format(env.load))
                print('action: {}'.format(action))
            if is_done:
                print('Iteration: {}, Score: {}'.format(i, reward_history[i]))
                break
        best = max(best, reward_history[i])
        returns = compute_returns(rewards)

        log_probs = torch.cat(saved_logprobs)
        returns = torch.cat(returns).detach()
        values = torch.cat(saved_values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return reward_history

def reinforce(rt_table, new_req_seq, n_episode=1000, gamma=1):
    env = Env_static(rt_table, new_req_seq, 90, 5)
    n_input = env.observation_space
    action_space = env.action_space
    hl_size = 128
    ri = Reinforce(hl_size, 5, 5)
    optimizer = optim.Adam(ri.parameters())
    reward_history = np.zeros(n_episode)
    best = -1
    for i in range(n_episode):
        state, load = env.reset()
        saved_logprobs = []
        saved_values = []
        rewards = []
        for t in range(1000):
            action, _, dist = get_action(state, load, ri)
            log_prob = dist.log_prob(action).unsqueeze(0)
            saved_logprobs.append(log_prob)
            state, load, reward, is_done = env.step(action.item())
            rewards.append(torch.tensor([reward], dtype=torch.float))
            reward_history[i] += reward
            if i == n_episode - 1:
                print('time: {}, state: {}'.format(env.time, env.state))
                print('load: {}'.format(env.load))
                print('action: {}'.format(action))
            if is_done:
                print('Iteration: {}, Score: {}'.format(i, reward_history[i]))
                break
        best = max(best, reward_history[i])
        returns = compute_returns(rewards)

        log_probs = torch.cat(saved_logprobs)
        returns = torch.cat(returns).detach()
        loss = -(log_probs * returns).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(sum(p.numel() for p in ri.parameters() if p.requires_grad))

    return reward_history



def main(n_episode= 1000, gamma=1):
    curr_status = System_Status()
    read_layer(curr_status, "vgg16_titanx_default_pred.txt")
    curr_status.group_batch[0] = 3
    curr_status.group_batch[1] = 1
    curr_status.group_batch[2] = 5
    curr_status.group_batch[3] = 0
    curr_status.group_batch[4] = 2
    rt_table = np.array(curr_status.group_batch_matrix)
    f = open('request.txt','r')
    new_req_seq = []
    ac_reward = ac_gae(rt_table, new_req_seq)

    ri_reward = reinforce(rt_table, new_req_seq)
    np.save('ac_static.npy', ac_reward)
    np.save('ri_static.npy', ri_reward)
    baseline_1 = -0.35
    baseline_2 = -0.65
    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=5)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.grid(color='gray', alpha = 0.6, linestyle='-', linewidth=1)
        plt.plot(ac_reward, label='GAE Actor-Critic')
        plt.plot(ri_reward, label='REINFORCE')
        plt.hlines(baseline_1, 0, 999, color = 'gray', ls='dashed', label='always batching')
        plt.hlines(baseline_2, 0, 999, color = 'green', ls='dashed', label='no batching')
        plt.xlabel('Episode', fontsize=18)
        plt.ylabel('Total Rewards', fontsize=18)
        plt.legend()
        plt.savefig('tex/figures/runing_time.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()



