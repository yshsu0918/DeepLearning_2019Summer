from lab3_lib import *
from Sdataloader import *
from model import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime
import os
MAX_LENGTH = 20

def lab3_test_dataset():
    original = ['abandon', 'abet', 'begin', 'expend', 'sent', 'split', 'flared', 'functioning', 'functioning', 'healing']
    x = [ str2tensor(ss) for ss in original ]
    xc = [0, 0, 0, 0, 3, 0, 3, 2, 2, 2]     
    y = ['abandoned', 'abetting', 'begins', 'expends', 'sends', 'splitting', 'flare', 'function', 'functioned', 'heals']    
    yc = [3, 2, 1, 1, 1, 2, 0, 0, 3, 1]
    
    return original, x, xc , y ,yc


def test_bleu(cvae):
    cvae.eval()
    cvae.E.eval()
    cvae.D.eval()
    cvae.R.eval()
    
    test_dataloader = DataLoader(
        dataset = lab3_test_dataset(), 
        batch_size=1, 
        shuffle=False, 
        num_workers=1
    )
    result = []
    with torch.no_grad():
        originals, x, xc , y ,yc = lab3_test_dataset()
        for test_idx in range(len(originals)):
            c = cvae.condition( xc[test_idx] )
            target_c = cvae.condition( yc[test_idx] )

            #lookup('c',c,content=True)
            #lookup('target_c',c,content=True )

            encoder_hidden = torch.cat( (cvae.initHidden(), c), dim = 2)
            encoder_output = 0
            input_length = x[test_idx].size(0)
            for i in range(input_length):
                encoder_output, encoder_hidden = cvae.E( x[test_idx][i] , encoder_hidden)
            
            mu, logvar, latent_tensor =  cvae.R(encoder_hidden)

            output_string = ''
            target_length = input_length 
            decoder_input = torch.LongTensor([SOS_token]).to(device)
            decoder_hidden = torch.cat( (cvae.latent2decoder_hidden(latent_tensor) , target_c) , dim =2)
            decoder_outputs = [ torch.zeros(1,28).to(device) ] * 20

            for i in range( 20 ):
                decoder_input = decoder_input.detach()
                decoder_output , decoder_hidden = cvae.D(decoder_input , decoder_hidden)
                idx = torch.argmax(decoder_output.view(1, -1))
                output_string += num2alpha[ idx.item() ]
                decoder_outputs[i] = decoder_output                
                decoder_input = idx
                if( num2alpha[ idx.item() ] == 'EOS' ):
                    break

            result.append(( originals[test_idx] , y[test_idx] , xc[test_idx], yc[test_idx] , output_string))
            
    bleu_total = 0 
    for x in result:
        bleu = compute_bleu( x[1].replace('EOS', ''), x[4].replace('EOS', ''))
        bleu_total += bleu
        print(x, bleu)
    print("Average bleu score = ", bleu_total / len(result)) 


from pympler.asizeof import asizeof

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def detect_mem(varname, var):
    print(varname, get_size(var), asizeof(var))


def train( cvae , max_epoch = 100, print_every=1, learning_rate=0.01, losss=[]):
    pairs = load_data() #memory leak?
    def prepare_epoch_data(pairs):
        random.shuffle(pairs)
        training_inputs = [ pairs[i][0] for i in range(len(pairs))]
        training_conditions = [ pairs[i][1] for i in range(len(pairs))]
        return training_inputs, training_conditions

    cvae_optimizer = optim.SGD(cvae.parameters(), lr = learning_rate)
    decoder_criterion = nn.CrossEntropyLoss()
    bleu_max = -1
    
    for epoch in range( max_epoch) :
        test_bleu(cvae)
        start = time.time()
        
        encoder_loss_total = 0
        cvae_total = 0
        bleu_total = 0 
    
        training_inputs, training_conditions = prepare_epoch_data(pairs)
        output_string = ''
        for iter in range( len(training_inputs) ):
            cvae.train()
            cvae_optimizer.zero_grad()
            
            dataset_idx = iter 
            
            input_tensor = Variable( str2tensor(training_inputs[dataset_idx]) )
            condition = training_conditions[dataset_idx]
            
            decoder_outputs , mu , logvar , output_string = cvae( input_tensor, condition)
        
            kld_w = KLDW(iter, sys.argv[4])
            encoder_loss = KL_loss(mu , logvar)
            decoder_loss = decoder_criterion( decoder_outputs, input_tensor)
            loss_cvae = decoder_loss + (kld_w * encoder_loss)

            loss_cvae.backward()        
            
            cvae_optimizer.step()

            encoder_loss_total += encoder_loss
            cvae_total += loss_cvae.item()
            bleu_total += compute_bleu( training_inputs[dataset_idx].replace('EOS',''), output_string.replace('EOS', ''))

        if epoch % print_every == 0:
            bleu_avg = bleu_total / len(training_inputs)
            cvae_avg = cvae_total / len(training_inputs)
            encoder_loss_avg = encoder_loss_total / len(training_inputs)
            
            print('epoch ', epoch , 'time: ', timeSince(start))
            print('loss', cvae_avg , 'bleu: ', bleu_avg)           
            print( 'Sample out:\n',training_inputs[dataset_idx] ,'->', output_string.replace('EOS', '') )
            
            losss.append( (cvae_avg, encoder_loss_avg,  bleu_avg) )
            
            if bleu_avg > bleu_max:
                bleu_max = bleu_avg
                print('Save model...')
                save_var(losss, sys.argv[1], 'loss')
                save_cvae(cvae, os.path.join(sys.argv[1], 'cvae_best') )
                print('Save model complete ...')

def main():
    max_epoch = int(sys.argv[3])
    cvae = CVAE()
    cvae.to(device)
    need_load = 1 if len(sys.argv) == 6 else 0
    if need_load:
        print('loading...')
        load_cvae(cvae, os.path.join(sys.argv[1], 'cvae_best' ))
        with open( os.path.join(sys.argv[1] ,'loss.pickle'), 'rb') as file:
            losss = pickle.load(file)
            file.close()
        os.rename(os.path.join(sys.argv[1] ,'loss.pickle'), 
        os.path.join(sys.argv[1] ,'loss'+ datetime.datetime.now().__str__().replace(' ','-')+'.pickle') )
        train( cvae , max_epoch , learning_rate = float(sys.argv[2]) , losss = losss)

    else:
        train( cvae , max_epoch,  learning_rate = float(sys.argv[2])) 


if __name__ == '__main__':
    if len(sys.argv) >= 5:
        main()

    else:
        print('Usage')
        print('TRAIN: python3 lab3_Sam.py [checkpoint_path] [learning_rate] [iter] [KLWD M/C] [opt_load 1]')
        print('TEST : python3 lab3_Sam.py [checkpoint_path] [iter (load weight file)]')
        



