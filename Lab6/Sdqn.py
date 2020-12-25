"""DLP DQN Lab"""
__author__ = 'chengscott'
__copyright__ = 'Copyright 2019, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time
import numpy as np
import gym
import torch
import torch.nn as nn


class ReplayMemory:
  def __init__(self, capacity):
    self._buffer = deque(maxlen=capacity)

  def __len__(self):
    return len(self._buffer)

  def append(self, *transition):
    # (state, action, reward, next_state, done)
    self._buffer.append(tuple(map(tuple, transition)))

  def sample(self, batch_size=1):
    return random.sample(self._buffer, batch_size)


class DQN(nn.Module):
  def __init__( self, state_dim=4, action_dim=2, hidden_size=24):
    super(DQN, self).__init__()
    
    self.state_dim = state_dim   
    self.action_dim = action_dim
    self.hidden_size = hidden_size
    self.net = nn.Sequential(
        nn.Linear(self.state_dim, self.hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_size, self.hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_size, self.action_dim, bias=False),
        )
  
  def forward(self, observation):
    return self.net(observation)
    
def select_action(epsilon, state, action_dim=2):
  if np.random.rand() < epsilon:
    action = np.random.randint( action_dim )
  else:
    action_ = eval_net(state)
    lookup('action_', action_, content = True)
    action = torch.argmax(action_).item()
    lookup('action', action, content = True)
  return action

def lookup(comment,var,show = False, content=False):
  if not show:
    return 
  try:
    print(comment, var.shape)
    if content:
      print(var)
  except:
    print(comment, type(var))
    if content:
      print(var)

def update_eval_network():
  def transitions_to_tensors(transitions, device=args.device):
    """convert a batch of transitions to tensors"""
    return (torch.Tensor(x).to(device) for x in zip(*transitions))

  # sample a minibatch of transitions
  transitions = memory.sample(args.batch_size)
  state, action, reward, next_state, done = transitions_to_tensors(transitions)

  
  q_eval_ = eval_net( state ).to( args.device ) 
  q_eval = torch.max(q_eval_, dim=1)[0]
  q_target_ = target_net(next_state).to(args.device).detach()
  q_target = torch.max(q_target_, dim=1)[0].detach()
  q_target[np.argwhere(done == 1).reshape(-1)] = 0.0

  #lookup('q_target_',q_target_, show=True)
  #lookup('q_target',q_target, show=True)
  y =  reward.view(-1) + (args.gamma * q_target.view(-1))
  #lookup('y',y , show=True)
  #lookup('q_eval',q_eval , show=True)
  
  loss = criterion (q_eval, y)
  optimizer.zero_grad()
  loss.backward()
  nn.utils.clip_grad_norm_(eval_net.parameters(), 5)
  optimizer.step()  
'''
# pick data from buffer
pick_i = np.random.choice(self.buffer_size if self.buffer_count > self.buffer_size else self.buffer_count, size=batch_size)
x = self.buffer[pick_i, :]
# clear grad
self.eval_optimizer.zero_grad()
# only use state

# x[:, -1] done list
q_target_value = torch.max(q_target, dim=1)[0].detach()
q_target_value[np.argwhere(x[:,-1] == 1).reshape(-1)] = 0.0
y = q_eval.clone().detach()
# set y = r + \gamma max_a( Q_target(s', a) )
y[np.arange(batch_size), x[:, self.osize]]  = torch.tensor(x[:, self.osize+1]).float() + (gamma * q_target_value)

# get loss
loss = self.criterion(q_eval, y)
'''

def train(env):
  print('Start Training')
  total_steps, epsilon = 0, 1.
  test_score_max = -1
  for episode in range(args.episode):
    total_reward = 0  
    state = env.reset()
    for t in itertools.count(start=1):
      # select action
      if total_steps < args.warmup:
        action = env.action_space.sample()
      else:
        state_tensor = torch.Tensor(state).to(args.device)
        action = select_action(epsilon, state_tensor)
        epsilon = max(epsilon * args.eps_decay, args.eps_min)
      # execute action
      next_state, reward, done, _ = env.step(action)
      # store transition
      memory.append(state, [action], [reward / 10], next_state, [int(done)])
      if total_steps >= args.warmup and total_steps % args.freq == 0:
        # update the behavior network
        update_eval_network()
      if total_steps % args.target_freq == 0:
        # TODO: update the target network by copying from the behavior network
        target_net.load_state_dict(eval_net.state_dict())
        #raise NotImplementedError

      state = next_state
      total_reward += reward
      total_steps += 1
      if done :
        print('Step: {}\tEpisode: {}\tTotal reward: {}\tEpsilon: {}'.format(
            total_steps, episode, total_reward, epsilon))
        break
    
    '''eval_net.eval()
    test_score = test(env, args.render)
    if test_score > test_score_max:
      torch.save(eval_net, args.model)
      test_score_max = test_score
      print('new high', test_score)
    eval_net.train()'''

  env.close()


def test(env, render, show=False):
  #print('Start Testing')
  epsilon = args.test_epsilon
  seeds = (20190813 + i for i in range(100))
  total_score = 0.0
  for seed in seeds:
    total_reward = 0
    env.seed(seed)
    state = env.reset()
    score = 0.0
    while(True):
      state_tensor = torch.Tensor(state).to(args.device)
      a = select_action(epsilon,state_tensor)
      next_state, reward, done, _ = env.step(a)
      score += reward
      state = next_state
      if show:
        #print('score : {:.0f}, action : {}, next_state : {}'.format(score, a, next_state))
        env.render()
      if done:
        print(score)
        total_score += score
        break
  return total_score/ 10



def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('-d', '--device', default='cpu')
  # network
  parser.add_argument('-m', '--model', default='cartpole_model')
  parser.add_argument('--restore', action='store_true')
  # train
  parser.add_argument('-e', '--episode', default=20000, type=int)
  parser.add_argument('-c', '--capacity', default=10000, type=int)
  parser.add_argument('-bs', '--batch_size', default=128, type=int)
  parser.add_argument('--warmup', default=10000, type=int)
  parser.add_argument('--lr', default=.0005, type=float)
  parser.add_argument('--eps_decay', default=.995, type=float)
  parser.add_argument('--eps_min', default=.01, type=float)
  parser.add_argument('--gamma', default=.99, type=float)
  parser.add_argument('--freq', default=4, type=int)
  parser.add_argument('--target_freq', default=1000, type=int)
  # test
  parser.add_argument('--test_epsilon', default=.001, type=float)
  parser.add_argument('--render', action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  # environment
  env = gym.make('CartPole-v1')
  # Intit eval target memory optimizer criterion
  eval_net = DQN().to(args.device)
  if not args.restore:
    # target network
    target_net = DQN().to(args.device)  
    target_net.load_state_dict(eval_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.RMSprop(eval_net.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    memory = ReplayMemory(capacity=args.capacity)
    train(env)
    torch.save(eval_net, args.model)
  #TEST
  eval_net = torch.load(args.model)
  eval_net.eval()
  test(env, args.render)
