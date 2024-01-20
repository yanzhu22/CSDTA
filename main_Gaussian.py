import os 
os.environ["CUDA_VISIBLE_DEVICES"]='1' 
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from dataset import SeqEmbDataset
from GaussianModel import GaussianNet, test
import joblib
import warnings  
warnings.filterwarnings('ignore')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

print(sys.argv)

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
SHOW_PROCESS_BAR = True
data_path = './CSDTA/data/'
seed = np.random.randint(33927, 33928)
path = Path(f'./CSDTA/runs/pkt2_{datetime.now().strftime("%m%d%H")}_{seed}')
device = torch.device("cuda")  # or torch.device('cpu')
            
max_seq_len = 1000  
max_smi_len = 150

batch_size = 256
n_epoch = 100
interrupt = None
save_best_epoch = 1
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True 

torch.manual_seed(seed)
np.random.seed(seed)

writer = SummaryWriter(path)
f_param = open(path / 'parameters.txt', 'w')

print(f'device={device}')
print(f'seed={seed}')
print(f'write to {path}')
f_param.write(f'device={device}\n'
          f'seed={seed}\n'
          f'write to {path}\n')
               

print(f'max_seq_len={max_seq_len}\n'
      f'max_smi_len={max_smi_len}')

f_param.write(f'max_seq_len={max_seq_len}\n'
      f'max_smi_len={max_smi_len}\n')

model = GaussianNet()

model = model.to(device)
f_param.write('model: \n')
f_param.write(str(model)+'\n')
f_param.close()

prots = joblib.load('./CSDTA/data/prot_emb_all.job')
drugs = joblib.load('./CSDTA/data/drug_emb_all.job')
data_loaders = {phase_name:
                    DataLoader(SeqEmbDataset(data_path, prots, drugs, phase_name,
                                         max_seq_len, max_smi_len),
                               batch_size=batch_size,
                               pin_memory=True,
                               num_workers=8,
                               shuffle=True #if phase_name=='training' else False
                                )
                for phase_name in ['training', 'validation','test', 'test105', 'test71']}
optimizer = optim.AdamW(model.parameters())

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, #5e-3, 
                                            epochs=n_epoch,
                                          steps_per_epoch=len(data_loaders['training']),
                                          )

loss_function = nn.MSELoss(reduction='mean')

start = datetime.now()
print('start at ', start)

model_file_name = 'wo_gaussian_best_model_dist'
early_stopping = EarlyStopping(patience=15, verbose=True, path=model_file_name)
    
best_epoch = -1
best_val_loss = 10000
for epoch in range(1, n_epoch + 1):
    tbar = tqdm(enumerate(data_loaders['training']), disable=not SHOW_PROCESS_BAR, total=len(data_loaders['training']))
    _lambda = epoch/n_epoch
    for idx, (*x, y) in tbar:
        model.train()

        for i in range(len(x)):
            x[i] = x[i].to(device)
            
        y = y.to(device)

        optimizer.zero_grad()
        output, vae_loss, _  = model(*x) 
        loss = (loss_function(output.view(-1), y.view(-1))) + (vae_loss)

        loss.backward() 
        optimizer.step()
        scheduler.step()
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item() :.3f}   ')

    performance = test(model, data_loaders['validation'], loss_function, device, False)
    val_loss = performance['loss']
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
model.load_state_dict(torch.load(  model_file_name ))

with open(path / 'result.txt', 'w') as f:
    f.write(f'best model found at epoch NO.{best_epoch}\n')
    for _p in [  'validation', 'test', 'test105', 'test71']:
        performance = test(model, data_loaders[_p], loss_function, device, SHOW_PROCESS_BAR)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}')
        f.write('\n')
        print()

print('training finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
