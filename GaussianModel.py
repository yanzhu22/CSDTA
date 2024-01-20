import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum
import numpy as np
from tqdm import tqdm
import metrics
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from entmax.activations import entmax15
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN
from dataset import PT_FEATURE_SIZE

CHAR_SMI_SET_LEN = 64

class ConcatELU(nn.Module):
    '''
    Activation function which applies ELU in both directions.
    '''
    def forward(self,x):
        return torch.cat([F.relu(x),F.relu(-x)],dim=1)
        
class GatedConv(nn.Module):
    def __init__(self,in_channels,hidden_channels,dilaSize=3):
        '''
        Create a two layer deep network for ResNet with input gate.
        Inputs:
            in_channels     - Number of input channels.
            hidden_channels - Number of hidden channels.
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels,kernel_size=3, padding=dilaSize,dilation=dilaSize),
            ConcatELU(),
            nn.Conv1d(2*hidden_channels, 2*in_channels, kernel_size=1),
        )

    def forward(self,x):
        out = self.net(x)
        val, gate = out.chunk(2,dim=1)
        x = x + val  * torch.sigmoid(gate)
        return x


class GatedResNet(nn.Module):
    def __init__(self,in_channels,hidden_channels,output_channels,num_layers=3,pool_size=2,dilaSize = 2):
        '''
        Creates GatedResNet using the previous modules.
        Inputs:
            in_channels     - Number of input channels.
            hidden_channels - Number of hidden channels.
            output_channels - Number of output channels (3K-1 * in_channels)
        '''
        super().__init__()
        
        layers = [nn.Conv1d(in_channels, hidden_channels,kernel_size=3,padding=dilaSize,dilation=dilaSize)]
        for _ in range(num_layers):
                layers += [
                    GatedConv(hidden_channels,hidden_channels,dilaSize),
                    nn.MaxPool1d(pool_size),          
                ]
             
        layers += [
            ConcatELU(),
            nn.Conv1d(2*hidden_channels,output_channels,kernel_size=3,padding=dilaSize,dilation=dilaSize),
        ]

        self.net = nn.Sequential(*layers)
         
    def forward(self,x):
        x = self.net(x.transpose(1,2))
        x = Rearrange('b n d -> b d n')(x)
        
        return x
    
class MuSigma(nn.Module):
    def __init__(self, input_dim):
        super(MuSigma, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU(),
        )
    def forward(self,  embedding):
        mu = self.layer1(embedding)
        sigma = self.layer2(embedding) + 1e-6 
        return mu, sigma    

class decoder(nn.Module):
    def __init__(self, init_dim, num_filters, k_size,size):
        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters , 1   * (init_dim - 3 * (k_size - 1))),
            nn.GELU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(1, num_filters, k_size, 1, 0),
            nn.GELU(),
            nn.ConvTranspose1d(num_filters, num_filters, k_size, 1, 0 ),
            nn.GELU(),
            nn.ConvTranspose1d(num_filters, num_filters, k_size, 1, 0 ),
            nn.GELU(),
        )
        self.gru = nn.GRU(num_filters, num_filters, batch_first=True, num_layers=2)
        self.layer2 =  nn.Sequential(
            nn.Linear(num_filters,size)
        )
        self.init_dim = init_dim
        self.num_filters = num_filters
        self.k_size = k_size 

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, 1, self.init_dim - 3 * (self.k_size - 1))
        x = self.convt(x)
        x = x.permute(0,2,1)
        x, _ = self.gru(x)
        x = self.layer2(x)
        return x

class GaussianNet(nn.Module):
    def __init__(self):
        
        super(GaussianNet, self).__init__()

        embed_dim = 256
        out_dim = 256
        hidden_dim = 256
        self.onehot_smi_net = GatedResNet( 384, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)
        self.onehot_prot_net = GatedResNet( 1024, hidden_dim, out_dim, num_layers=5, pool_size=2,dilaSize=1)

        self.musigma_prot = MuSigma(out_dim)
        self.musigma_drug = MuSigma(out_dim)

        self.transform = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1),
        )
        self.decoder1 = decoder(150, out_dim, 3, 64)
        self.decoder2 = decoder(1000, out_dim, 3, 25)
    
    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)


    def forward(self, seq, smi, seq_onehot, smi_onehot):

        proteinFeature_onehot = self.onehot_prot_net( seq) 
        compoundFeature_onehot = self.onehot_smi_net( smi )

        proteinFeature_onehot = Reduce('b n d -> b d', 'max')(proteinFeature_onehot)
        compoundFeature_onehot = Reduce('b n d -> b d', 'max')(compoundFeature_onehot)

        
        mu_drug, logvar_drug = self.musigma_drug(compoundFeature_onehot)
        mu_prot, logvar_prot = self.musigma_prot(proteinFeature_onehot)
         
        euclidean_dist = torch.square( mu_drug - mu_prot )

        w2_dist = euclidean_dist + torch.square(torch.sqrt( torch.exp(logvar_drug))-torch.sqrt( torch.exp(logvar_prot)))
   
        out = self.transform(w2_dist)

        drug_sample  = self.reparametrize(mu_drug, logvar_drug)
        target_sample  = self.reparametrize(mu_prot, logvar_prot)

        recon_drug = self.decoder1(drug_sample)
        recon_target = self.decoder2(target_sample)

        loss_drug = self.loss_f(recon_drug, smi_onehot, mu_drug, logvar_drug)
        loss_target = self.loss_f(recon_target, seq_onehot, mu_prot, logvar_prot)
        lamda = -1
        return out,  10**lamda * (loss_drug + 150 / 1000 * loss_target), w2_dist
    def  loss_f(self, recon_x, x, mu, logvar):
        cit = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        cr_loss = torch.sum(cit(recon_x.permute(0, 2, 1), x), 1)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return torch.mean(1e-2*cr_loss + KLD)




def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    loss_function = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat, _ , _ = model(*x)
            print(y_hat)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            # test_loss += loss_function(y_hat, y).item()

            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
        
    }
    with open('./CSDTA/t-test/'+str(test_loader)+'results.txt', 'w') as file:  
        for target, output in zip(targets, outputs):  
            file.write(f"{target:.2f} {output:.2f}\n")

    return evaluation
