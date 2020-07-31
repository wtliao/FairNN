import torch
import torch.utils.data as Data
from torch.autograd import Variable

import parameters as par
from Models import MyAutoencoder
from tensorboardX import SummaryWriter

EPOCH = par.AE_EPOCH
BATCH_SIZE = par.AE_BATCH_SIZE
LR = par.AE_LR
writer = SummaryWriter(par.AE_GRAPH_SAVE_PATH)

model = MyAutoencoder()
dummy_input = Variable(torch.rand(20, 107))
with SummaryWriter(comment='debias Autoencoder') as w:
    w.add_graph(model, (dummy_input,))
