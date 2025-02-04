import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np

from lab4_lib import *


class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

class Trainer:

  def __init__(self, G, FE, D, Q):

    self.G = G
    self.FE = FE
    self.D = D
    self.Q = Q

    self.batch_size = 100

    self.shortnames = ['G.pkl','F.pkl','D.pkl','Q.pkl']
    self.models = [self.G, self.FE, self.D , self.Q ]

  def save(self,epoch,PATH):     
    for i in range(4):
      filepath = os.path.join(PATH,self.shortnames[i]+'_'+str(epoch))
      torch.save(self.models[i].state_dict(), filepath)

  def load(self,epoch,PATH):
    for i in range(4):
      filepath = os.path.join(PATH,self.shortnames[i]+'_'+str(epoch))
      self.models[i].load_state_dict(torch.load(filepath))

  #def _noise_sample(self, dis_c,con_c, noise, bs, one2ten=False):
  def _noise_sample(self, dis_c, noise, bs, one2ten=False):
    
    if one2ten: #top 10 onehot is 0-9
      idxx = [i for i in range(10)]
      idxx.extend(np.random.randint(10, size=bs-10).tolist())
      idx = np.array(idxx)
    else:
      idx = np.random.randint(10, size=bs)
    
    c = np.zeros((bs, 10))
    c[range(bs),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    #con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    #lookup('noise', noise)
    #lookup('dis_c', dis_c)
    
    z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)

    return z, idx

  def train(self,PATH='./TEST', testflag = False):

    real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).cuda()
    label = torch.FloatTensor(self.batch_size, 1).cuda()
    dis_c = torch.FloatTensor(self.batch_size, 10).cuda()
    #con_c = torch.FloatTensor(self.batch_size, 2).cuda()
    noise = torch.FloatTensor(self.batch_size, 54).cuda()

    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    dis_c = Variable(dis_c)
    #con_c = Variable(con_c)
    noise = Variable(noise)

    criterionD = nn.BCELoss().cuda()
    criterionQ_dis = nn.CrossEntropyLoss().cuda()
    criterionQ_con = log_gaussian()

    optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

    dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    # fixed random variables
    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

    idx = np.arange(10).repeat(10)
    one_hot = np.zeros((100, 10))
    one_hot[range(100), idx] = 1
    fix_noise = torch.Tensor(100, 54).uniform_(-1, 1)
    losses = []
    for epoch in range(100):
      
      if epoch % 1 == 0:
        PATH = './TEST'
        PATH_weight = './TEST/weight'
        self.save(epoch,PATH_weight)
        #self.generate_image(PATH, 'img_'+str(epoch)+'.png')
        with open( os.path.join(PATH, 'loss_epoch_'+str(epoch) ) , 'w') as f:
            for x in losses:
                f.write( x.__str__() + '\n')
            f.close()
        print('Epoch ', epoch, ' save complete...')      
      for num_iters, batch_data in enumerate(dataloader, 0):
        # real part
        optimD.zero_grad()
        x, _ = batch_data
        bs = x.size(0) 
        real_x.data.resize_(x.size())
        label.data.resize_(bs, 1)
        dis_c.data.resize_(bs, 10)
        noise.data.resize_(bs, 54)
        
        real_x.data.copy_(x)
        fe_out1 = self.FE(real_x)
        probs_real = self.D(fe_out1)
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # fake part
        z, idx = self._noise_sample(dis_c, noise, bs)
        fake_x = self.G(z)
        fe_out2 = self.FE(fake_x.detach())
        probs_fake = self.D(fe_out2)
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake
        optimD.step()
        
        # G and Q part
        optimG.zero_grad()

        fe_out = self.FE(fake_x)
        probs_fake = self.D(fe_out)
        label.data.fill_(1.0)

        reconstruct_loss = criterionD(probs_fake, label)
        
        q_logits, q_mu, q_var = self.Q(fe_out)
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
        dis_loss = criterionQ_dis(q_logits, target)
        #con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1
        
        G_loss = reconstruct_loss + dis_loss
        G_loss.backward()
        optimG.step()

        if num_iters % 100 == 0:

          print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
            epoch, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy())
          )
          losses.append((D_loss.data.cpu().numpy(),G_loss.data.cpu().numpy()))
          
          noise.data.copy_(fix_noise)
          dis_c.data.copy_(torch.Tensor(one_hot))
          
          if testflag:
            self.test( noise, dis_c , './test_image.png')
          else:
            self.test(os.path.join(PATH, 'c2_'+ str(epoch)+'_'+str(num_iters)+'.png') , noise, dis_c )

  def test(self, noise, dis_c, PATH = './test_image.png'):

    z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)
    self.G.eval()
    x_save = self.G(z)
    save_image(x_save.data, PATH , nrow=10)

'''  def generate_image(self,PATH,fname,debug = False):
    dis_c = torch.FloatTensor(self.batch_size, 10).cuda()
    noise = torch.FloatTensor(self.batch_size, 54).cuda()
    dis_c = Variable(dis_c)
    noise = Variable(noise)        
    self.G.eval()
    z , idx = self._noise_sample(dis_c, noise, 100)
    fake_x = self.G(z)
    fake_one2ten = fake_x[0:10]
    save_image(fake_one2ten.data, os.path.join(PATH, fname), nrow=10)
    return z'''