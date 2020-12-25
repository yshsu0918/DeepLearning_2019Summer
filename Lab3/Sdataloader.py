
from lab3_lib import *

def load_data(train = True):
    result = []

    if train:
        with open('./train.txt','r') as f:
            s = f.read().replace('\n', ' ')
        phrase = s.split(' ')
        for i in range(len(phrase)):
            result.append( (phrase[i], i % num_tense) )
        return result
    
    else:
        #Not finish
        with open('./train.txt','r') as f:
            f.readlines()
    return result