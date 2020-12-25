import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply( x, 1.0 - x)

class MMM:
    #property in MMM
    #x,y current input/output data
    #W0,W1,W2 = First/Second/Third weight
    
    def __init__(self, raw_x, raw_y,iter=1,debug=1,learning_rate =0.1):
        print('init')
        self.init_weight()
        self.learning_rate = learning_rate
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.debug = debug
        self.iter=iter
        print('init finish')

    def init_weight(self):
        def generate_init_weight_matrix(m,n):
            x = np.random.randn(n, m) / np.sqrt(n / 2)
            Q = np.array(x).reshape(m,n)
            return Q

        self.W0 = generate_init_weight_matrix(4,2)
        self.W1 = generate_init_weight_matrix(4,4)
        self.W2 = generate_init_weight_matrix(1,4)
        
    def forward_feed(self):
        self.H1 = sigmoid( np.dot (self.W0 , self.x) )
        self.H2 = sigmoid( np.dot (self.W1, self.H1) )
        self.H3 = np.dot (self.W2 , self.H2)
        
        self.log('H1\n',self.H1)
        self.log('H2\n',self.H2)
        self.log('W0\n',self.W0)
        self.log('W1\n',self.W1)
        self.log('W2\n',self.W2)
        
        self.pred_y = self.H3[0][0]
        self.loss = 0.5*(self.pred_y - self.y)**2
        
        self.log(self.pred_y,self.y,self.loss)
    
    def backward(self):
        
        # ------------------------------------
        #    output layer to hidden layer2
        # ------------------------------------
        
        delta_W =  (2 * (self.pred_y-self.y) * self.H2 ).reshape(1,4)
        delta_H2 =  (2 * (self.pred_y-self.y) * self.W2).reshape(4,1)
        
        self.log('delta_W: \n', delta_W)
        self.log('delta_H2: \n', delta_H2)
        
        self.W2 -= self.learning_rate * delta_W
        # --------------- ---------------------
        #    hidden layer2 to hidden layer1
        # ------------------------------------
        
        delta_S_H2 = derivative_sigmoid(self.H2)
        '''
        delta_WW1 = (delta_H2 * delta_S_H2 * self.H1[0]).reshape(1,4)
        delta_WW2 = (delta_H2 * delta_S_H2 * self.H1[1]).reshape(1,4)
        delta_WW3 = (delta_H2 * delta_S_H2 * self.H1[2]).reshape(1,4)
        delta_WW4 = (delta_H2 * delta_S_H2 * self.H1[3]).reshape(1,4)
        
        delta_WW = np.concatenate( (delta_WW1.T, delta_WW2.T, delta_WW3.T, delta_WW4.T), axis = 1 )
        '''
        
        delta_WW = self.H1.T * (delta_S_H2 *delta_H2)

        delta_H1 = np.dot( self.W1 ,  (delta_H2*delta_S_H2) )

        self.log('delta_H1 \n', delta_H1)
        
        self.W1 -= self.learning_rate * delta_WW        
        # ------------------------------------
        #    hidden layer1 to input layer
        # ------------------------------------ 
        
        delta_S_H1 = derivative_sigmoid(self.H1)
                
        delta_WWW =  delta_H1 * delta_S_H1 * self.x.reshape(1,2)
        self.log('delta_WWW: \n', delta_WWW)
        
        self.W0 -=  self.learning_rate * delta_WWW


        
    def log(self,*args):
        if self.debug == 1:
            print(*args)
        else:
            pass
    
    def learn(self):
        for epoch in range(self.iter):
            c = 0
            for i in range(len(self.raw_x)):
                self.x = self.raw_x[ i ].reshape(2,1)
                self.y = self.raw_y[ i ]            
                self.forward_feed()
                self.backward()
                c += self.loss
            if epoch % (self.iter/10) == 0:
                print('epoch', epoch , ' loss : ', c/len(self.raw_x))
            
    def test(self,test_x,test_y):
        c = 0
        pred_y = []
        for i in range(len(test_x)):
            self.x = test_x[ i ].reshape(2,1)
            self.y = test_y[ i ]            
            self.forward_feed()
            c += self.loss
            #print( test_x[i], round(self.pred_y) , test_y[ i ], self.loss)
            #pred_y.append( abs(1-round(self.pred_y)) )
            pred_y.append( round(self.pred_y) )
            print(self.pred_y)
        #print('test result:\n avg_loss=', c/len(test_x))
        
        show_result(test_x,test_y,pred_y)
        
def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append( [ pt[0], pt[1] ])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append( [0.1*i,0.1*i] )
        labels.append( 0 )
        if 0.1*i == 0.5:
            continue
        
        inputs.append( [0.1*i, 1 - 0.1*i] )
        labels.append( 1 )
        
    return np.array(inputs), np.array(labels).reshape(21,1)


def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot( x[i][0], x[i][1], 'ro')
        else:
            plt.plot( x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot( x[i][0], x[i][1], 'ro')
        else:
            plt.plot( x[i][0], x[i][1], 'bo')
    plt.show()

if __name__ == '__main__':
    TEST = [generate_linear]
    
    for generate_func in TEST:
        x,y = generate_func()
        NET = MMM(x,y,debug=0,iter=1000,learning_rate=0.1)
        NET.learn()
        x,y = generate_func()
        NET.test(x,y)
