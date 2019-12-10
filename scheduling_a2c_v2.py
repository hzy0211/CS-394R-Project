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
from randrequests import genNewReq
from baseline import always_batching, no_batching
import seaborn as sns

class Env:
    def __init__(self, curr_status, new_req_seq, max_job, n_layers=38):
        self.n_layers = n_layers
        self.rt_table = np.array(curr_status.group_batch_matrix)
        assert self.rt_table.shape[1] == self.n_layers
        self.max_job = max_job
        self.curr_status = curr_status
        self.new_req_seq = new_req_seq
        self.reset()

    def reset(self):
        self.time = 0
        self.time_till_last = 0
        self.state = np.zeros(self.n_layers)
        self.load = np.zeros(self.n_layers)
        self.job_counter = 0
        self.wait_job = 0
        self.is_done = False
        self.observation_space = self.n_layers
        self.action_space = self.n_layers
        for i in range(self.n_layers):
            if self.curr_status.group_batch[i] > 0:
                self.load[i] = 1
            else:
                self.load[i] = 0
            self.state[i] = self.curr_status.group_batch[i]
        return self.state, self.load

    def new_request(self):
        while self.wait_job > 0 and self.state.sum() < self.max_job:
            self.state[0] += 1
            self.load[0] = 1
            self.wait_job -= 1

        while self.job_counter < NUM_NEW_REQUEST:           
            if self.new_req_seq[self.job_counter] <= self.time and self.state.sum() < self.max_job:
                self.state[0] += 1
                self.load[0] = 1
                self.job_counter += 1
            elif self.new_req_seq[self.job_counter] <= self.time and self.state.sum() >= self.max_job:
                self.wait_job += 1
                self.job_counter += 1
            else:
                break

    def step(self, action):
        layer_select = action
        running_load = self.state[layer_select]
        n_jobwaiting = self.state.sum() + self.wait_job
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
        if self.state.sum() == 0 and self.job_counter >= NUM_NEW_REQUEST:
            self.is_done = True
        if self.state.sum() == 0 and self.job_counter < NUM_NEW_REQUEST:
            self.time = max(self.time, self.new_req_seq[self.job_counter])
        return self.state, self.load, reward, self.is_done

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

def get_action(state, load, actor_critic):
    state = torch.tensor(state).float()
    load = torch.tensor(load).float()
    action_probs, v_s= actor_critic(state, load)
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

def compute_time(start_idx, end_idx, batch_size, curr_status):
    time = 0
    for i in range(start_idx, end_idx, 1):
        time += curr_status.group_batch_matrix[batch_size-1][i]
    return time

def rl(curr_status, new_req_seq, n_episode=300, gamma=1):
    env = Env(curr_status, new_req_seq, NUM_NEW_REQUEST, GROUP_NUM)
    n_input = env.observation_space
    action_space = env.action_space
    hl_size = 128
    ac = Actor_Critic(hl_size, 5, 5)
    optimizer = optim.Adam(ac.parameters())
    reward_history = np.zeros(n_episode)
    best = -1
    output = []
    for i in range(n_episode):
        state, load = env.reset()
        saved_logprobs = []
        saved_values = []
        rewards = []
        for t in range(1000):
            env.new_request()
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
                output.append(action)
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
def initialize_status(curr_status):
    curr_status.group_batch[0] = 3
    curr_status.group_batch[1] = 1
    curr_status.group_batch[2] = 5
    curr_status.group_batch[3] = 0
    curr_status.group_batch[4] = 2
    return curr_status



def main():
    curr_status = System_Status()
    read_layer(curr_status, "vgg16_titanx_default_pred.txt")
    curr_status.group_batch[0] = 3
    curr_status.group_batch[1] = 1
    curr_status.group_batch[2] = 5
    curr_status.group_batch[3] = 0
    curr_status.group_batch[4] = 2
    rl_reward = np.zeros(13)
    nb_reward = np.zeros(13)
    ab_reward = np.zeros(13)
    number_users = np.array([1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    for i, n in enumerate(number_users):
        new_req_seq = genNewReq(n)
        rl_reward[i] = -np.mean(rl(curr_status, new_req_seq, 200)) / 61
        curr_status = initialize_status(curr_status)
        nb_reward[i] = no_batching(curr_status, new_req_seq) / 61
        curr_status = initialize_status(curr_status)
        ab_reward[i] = always_batching(curr_status, new_req_seq) / 61
        curr_status = initialize_status(curr_status)

    np.save('ac_streaming.npy', rl_reward)
    np.save('nb_streaming.npy', nb_reward)
    np.save('ab_streaming.npy', ab_reward)

    with sns.axes_style('white', {'legend.frameon': True}):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=5)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.grid(color='gray', alpha = 0.6, linestyle='-', linewidth=1)
        plt.plot(number_users, rl_reward, label='GAE Actor-Critic')
        plt.plot(number_users, ab_reward, label='always batching')
        plt.plot(number_users, nb_reward, label='no batching')
        plt.xlabel('Number of users', fontsize=18)
        plt.ylabel('Average waiting time', fontsize=18)
        plt.legend()
        plt.savefig('streaming.pdf', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()



