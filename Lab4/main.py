from model import *
from trainer import Trainer
import os
import argparse
import torch
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--load', default=0, type=int)
args = parser.parse_args()

def load_model(d,f,q,g):
  fnames = ['D.pkl','F.pkl','Q.pkl','G.pkl']
  for i, model in enumerate([d,f,q,g], 0):
    PATH = os.path.join('./TEST/weight', fnames[i] + '_'+ str(args.load))
    model.load_state_dict(torch.load(PATH))




if __name__ == '__main__':
  fe = FrontEnd()
  d = D()
  q = Q()
  g = G()
  for i in [fe, d, q, g]:
    i.cuda()
  if args.load != 0:
    print('testmode')
    load_model(d,fe,q,g)
    trainer = Trainer(g, fe, d, q)
    idx = np.arange(10).repeat(10)
    one_hot = np.zeros((100, 10))
    one_hot[range(100), idx] = 1    
    noise = torch.Tensor(100, 54).uniform_(-1, 1).cuda()
    dis_c = torch.Tensor(one_hot).cuda()
    #trainer.train(testflag = True)
    trainer.test(noise, dis_c)
  else:
    for i in [fe, d, q, g]:
      i.cuda()
      i.apply(weights_init)
    trainer = Trainer(g, fe, d, q)
    trainer.train()
  