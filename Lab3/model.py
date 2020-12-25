
from lab3_lib import *


class EncoderRNN(nn.Module):
    def __init__(self, word_size=28, hidden_size = 256, latent_size=32):
        super(EncoderRNN, self).__init__()
        self.word_size = word_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size
        
        self.word_embedding = nn.Embedding(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        # get (seq, 1, hidden_size)
        x = self.word_embedding(inputs).view(-1, 1, self.hidden_size)
        # get (seq, 1, hidden_size), (1, 1, hidden_size)
        outputs, hidden = self.gru(x, hidden )
        return outputs, hidden



class Reparameter(nn.Module):
    def __init__(self, encoder_output_size =256, latent_size = 32):
        super(Reparameter, self).__init__()
        self.latent_size = latent_size
        self.mu = nn.Linear(encoder_output_size, latent_size)
        self.log_var = nn.Linear(encoder_output_size, latent_size)

    def forward(self, encoder_output):
        mu = self.mu(encoder_output)
        log_var = self.log_var(encoder_output)
        std = torch.exp(0.5 * log_var)
        #eps = torch.randn_like(std)
        eps = torch.randn(1,1,self.latent_size,device=device)
        return mu, log_var, mu + eps*std



class DecoderRNN(nn.Module):
    def __init__(self, word_size=28, hidden_size=256, latent_size=32, condition_size=8):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_size = word_size
        
        self.latent_to_hidden = nn.Linear( latent_size+condition_size, hidden_size )
        
        self.word_embedding = nn.Embedding(word_size, hidden_size)# logic bug maybe...
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, word_size)


    def forward(self, x, hidden ):
        # get (1, 1, hidden_size)
        x = self.word_embedding(x).view(1, 1, self.hidden_size)
        # get (1, 1, hidden_size) (1, 1, hidden_size)
        output, hidden = self.gru(x, hidden )
        # get (1, word_size)
        output = self.out(output[0])
        return output, hidden

class CVAE(nn.Module):
    def __init__(self, word_size=28, hidden_size=256, latent_size=32, num_condition = 4, condition_size=8):
        super(CVAE, self).__init__()
        
        self.word_size = word_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size                
        
        self.E = EncoderRNN()
        self.D = DecoderRNN()
        self.R = Reparameter()
        self.condition_embedding = nn.Embedding(num_condition, condition_size)
        self.latent2decoder_hidden = nn.Linear(latent_size, self.hidden_size - self.condition_size )
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size - self.condition_size,device=device)
    def condition(self, c):
        c = torch.LongTensor([c]).to(device)
        return self.condition_embedding(c).view(1,1,-1)
    
    def forward(self , input_tensor , condition_scalar, test = False):

        c = self.condition( condition_scalar )
        encoder_hidden = torch.cat( (self.initHidden(), c), dim = 2)
        encoder_output = 0
        input_length = input_tensor.size(0)
        for i in range(input_length):
            encoder_output, encoder_hidden = self.E( input_tensor[i] , encoder_hidden)
        
        mu, logvar, latent_tensor =  self.R(encoder_hidden)

        #teacher_forcing_ratio = float(sys.argv[2]) 
        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        output_string = ''
        target_length = input_length 
        decoder_input = torch.LongTensor([SOS_token]).to(device)

        decoder_hidden = torch.cat( (self.latent2decoder_hidden(latent_tensor) , c) , dim =2)
        decoder_outputs = [ torch.zeros(1,28).to(device) ]*20

        for i in range( target_length ):
            decoder_input = decoder_input.detach()

            decoder_output , decoder_hidden = self.D(decoder_input , decoder_hidden)
            
            idx = torch.argmax(decoder_output.view(1, -1))
            output_string += num2alpha[ idx.item() ]
            decoder_outputs[i] = decoder_output

            if use_teacher_forcing:
                #print('#',end='')
                decoder_input = input_tensor[ i ]
            else: 
                #print('@',end='')
                #decoder_input = topi.squeeze().detach()
                decoder_input = idx
                if( num2alpha[ idx.item() ] == 'EOS' ):
                    break

        
        decoder_outputs = torch.cat(decoder_outputs[:input_length], dim=0)
        #print('after cat', decoder_outputs.size(0), input_tensor.size(0))

        return decoder_outputs , mu , logvar , output_string