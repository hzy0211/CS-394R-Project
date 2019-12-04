import numpy as np
from itertools import count
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math
from utils import *
from Request import Request
from Model_Parameters import Model_Parameters
from System_Status import System_Status
from Read_Layer import Read_Layer


class Env:
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
        self.state[0] = 3
        self.state[1] = 1
        self.state[2] = 5
        self.state[3] = 0
        self.state[4] = 2

    def new_request(self):
        while (self.job_counter < NUM_NEW_REQUEST):           
            if self.new_req_seq[self.job_counter] <= self.time and self.state.sum() < self.max_job:
                self.state[0] += 1
                self.job_counter += 1
            else:
                break

    def step(self, action):
        layer_select = action
        running_load = self.state[layer_select]
        # print('# job running: {}'.format(running_load))
        n_jobwaiting = self.state.sum()
        self.state[layer_select] = 0
        # print('# job waiting: {}'.format(n_jobwaiting))
        if layer_select + 1 < self.n_layers:
            self.state[layer_select + 1] += running_load
        running_time = self.rt_table[max(int(running_load - 1), 0), layer_select]
        # print('running_time: {}'.format(running_time))
        self.time += running_time
        # print('time till last: {}'.format(self.time_till_last))
        reward = -running_time * n_jobwaiting
        #if running_load == 0:
        #s    reward = -100
        if self.state.sum() == 0:
            self.is_done = True
        #if self.job_counter < NUM_NEW_REQUEST and self.state.sum() == 0:
        #    self.time = max(self.time, self.new_req_seq[self.job_counter])
        #if self.job_counter == NUM_NEW_REQUEST and self.state.sum() == 0:
        #    self.is_done = True
        return self.state, reward, n_jobwaiting, self.is_done


class Policy(nn.Module):
    def __init__(self, hl_size, n_input, action_space):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_input, hl_size)
        self.affine2 = nn.Linear(hl_size, action_space)

        self.saved_log_probs = []
        self.rewards = []
        self.size_history = []
        self.action_history = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class REINFORCE:
    def __init__(self, hl_size, n_input, action_space):
        self.model = Policy(hl_size, n_input, action_space)
        self.base_line = None

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.model(state)
        action_probs = np.array(action_probs.detach())
        np_state = np.array(state.detach())
        for i in range(len(np_state[0])):
            if (np_state[0][i] == 0.0):
                action_probs[0][i] = 0
        action_probs = action_probs / np.sum(action_probs)
        action_probs = torch.from_numpy(action_probs).float().unsqueeze(0)
        action_probs.requires_grad = True
        m = Categorical(action_probs)
        action = m.sample()
        #print('log_prob:{}'.format(m.log_prob(action)))
        self.model.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def set_baseline(self, n_action, n_size):
        self.base_line = np.zeros((n_action, n_size + 1))
        self.base_line_n = np.zeros_like(self.base_line)

    def update(self, gamma):
        R = 0
        policy_loss = []
        returns = []
        optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        for r, action, size in zip(self.model.rewards[::-1], self.model.action_history[::-1], self.model.size_history[::-1]):
            #size = int(size)
            #self.base_line[action, size] += (r - self.base_line[action, size]) / (self.base_line_n[action, size] + 1)
            #self.base_line_n[action, size] += 1
            R = r + gamma * R + 0.3 #- self.base_line[action, size]
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        for log_prob, R in zip(self.model.saved_log_probs, returns):
            # policy_loss.append(-log_prob * R)
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        # print('loss:{}'.format(policy_loss))
        policy_loss.backward()
        optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_log_probs[:]

def read_layer(curr_status, data_file):
    #print("batch size: ",BATCH_SIZE)
    #print("num_shared_layers: ",curr_status.num_shared_layers)
    #print("group_num_t: ",group_num_t, "group_num_shared: ",GROUP_NUM_SHARED)
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
    #print('first batch: ',curr_status.batch_matrix[0][:])
    #print("first layer: ",curr_status.batch_matrix[:][0])
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
            #print("curr_sum: ",curr_sum,"group_num: ",group_num,"i: ",i, "num_shared_layers: ", curr_status.num_shared_layers)
            assert (group_num>=0)
            assert (curr_sum>=0)
            j += 1

def main(n_episode=10000, gamma=1):
    curr_status = System_Status()
    read_layer(curr_status, "vgg16_titanx_default_pred.txt")
    rt_table = np.array(curr_status.group_batch_matrix)
    f = open('request.txt','r')
    new_req_seq = []
    for i in f.readline().split():
        new_req_seq.append(float(i))
    f.close()
    env = Env(rt_table, new_req_seq, 90, 5)
    #env.seed(0)
    #torch.manual_seed(0)
    n_input = env.observation_space
    action_space = env.action_space
    hl_size = 128
    agent = REINFORCE(hl_size, n_input, action_space)
    agent.set_baseline(4, 6)
    eps = 1e-6
    for i in range(n_episode):
        env.reset()
        episode_reward = 0
        print(i)
        for t in range(10000):
            #env.new_request()
            action = agent.select_action(env.state)
            if i == n_episode - 1:
                print('time: {}, state: {}'.format(env.time, env.state))
                print('action: {}'.format(action))
            state, reward, n_jobwaiting, is_done= env.step(action)
            agent.model.rewards.append(reward)
            agent.model.action_history.append(action)
            agent.model.size_history.append(n_jobwaiting)
            episode_reward += reward
            if is_done:
                print('done: {}'.format(t))
                break
        agent.update(gamma)
        print('episode:{}'.format(episode_reward))


if __name__ == '__main__':
    main()