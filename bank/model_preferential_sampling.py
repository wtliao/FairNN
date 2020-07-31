import os
import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import parameters as par
from Dataset_maker import Dataset_maker
from Custom_Loss_function import Loss_fairness_regularization
from functions.utils import dataset_splite, Preferential_sampling
from functions.classifier_metrics import accuracy_calculate_gender_ae, EO_evaluation_ae

URL = par.BANK_URL
TEST_DATA_SAVE = par.P_TEST_PATH
MODEL = par.MODEL_LOAD
SAVE_PATH = par.P_MODEL_SAVE
PLOT = par.P_PLOT
EPOCH = par.P_EPOCH


def P_sampling(train_loader, model):
    for i, (data, label) in enumerate(train_loader):
        data_tensor = Variable(data)
        label_true = Variable(label)
        # ============== forward ==================
        _, _, label_pred = model(data_tensor)
        # ============= preferential sampling ==========
        train_set_new = Preferential_sampling(label_pred, label_true, data_tensor)
        train_dataset = Data.TensorDataset(train_set_new[:, 0:50], train_set_new[:, 50:])
        train_loader_s = Data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True,
                                       num_workers=0)

    return train_loader_s


def P_sampling_tuning(train_loader, model, clf_loss, optimizer):
    for i, (data, label) in enumerate(train_loader):
        data_tensor = Variable(data)
        label_true = Variable(label)
        # ============== forward ==================
        _, _, label_pred = model(data_tensor)
        loss_ce, _ = clf_loss(label_pred, label_true)
        # ============== backward =================
        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

    return loss_ce, model


def P_sampling_val(val_loader, val_db, model, clf_loss):
    # ======================== validation =========================
    for i, (data_val, label_val) in enumerate(val_loader):
        data_tensor_val = Variable(data_val)
        label_true_val = Variable(label_val)
        _, _, label_pred_val = model(data_tensor_val)
        loss_ce, _ = clf_loss(label_pred_val, label_true_val)
        acc_val, _, _ = accuracy_calculate_gender_ae(val_db, model)
        _, EO= EO_evaluation_ae(val_db, model)

    return loss_ce, acc_val, EO


def test_save(dataloader, save_path):
    for i, (data, label) in enumerate(dataloader):
        array = np.hstack((data.numpy(), label.numpy()))
    df = pd.DataFrame(array, index=None)
    df.to_csv(save_path, index=False, sep=',')


if __name__ == '__main__':
    print('KLD:', par.RATIO_KLD)
    print('EO:', par.RATIO_EO)
    # =====make dataset=====
    MyDataset = Dataset_maker(URL, mode='c')
    print(len(MyDataset))
    train_db, val_db, test_db, _ = dataset_splite(MyDataset, 20002, 1000, 20002)

    train_loader = Data.DataLoader(dataset=train_db,
                                   batch_size=len(train_db),
                                   shuffle=True,
                                   num_workers=0)
    val_loader = Data.DataLoader(dataset=val_db,
                                 batch_size=len(val_db),
                                 shuffle=False,
                                 num_workers=0)
    test_loader = Data.DataLoader(dataset=test_db,
                                  batch_size=len(test_db),
                                  shuffle=False,
                                  num_workers=0)

    # ============ save test set ================
    test_save(test_loader, TEST_DATA_SAVE)
    # ============ make dir =====================
    if os.path.exists(SAVE_PATH) is False:
        os.mkdir(SAVE_PATH)
    if os.path.exists(PLOT) is False:
        os.mkdir(PLOT)

    # ============ P sampling ===================
    model = torch.load(MODEL)
    clf_loss = Loss_fairness_regularization()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=par.P_LR)
    writer = SummaryWriter(PLOT)

    # ============ sampling ====================
    train_loader = P_sampling(train_loader, model)
    # ============ fine tuning =================
    for iter in range(EPOCH):
        loss, model = P_sampling_tuning(train_loader, model, clf_loss, optimizer)
        loss_val, acc_val, EO_val = P_sampling_val(val_loader, val_db, model, clf_loss)
        # ============ log ===================
        print('epoch [{}/{}], train loss:{:.4f}, val loss:{:.4f}, val acc:{:.4f}'.format(iter + 1, EPOCH,
                                                                                        loss.data.item(),
                                                                                        loss_val.data.item(),
                                                                                        acc_val))
        writer.add_scalar('Train/Loss', loss.data.item(), iter + 1)
        writer.add_scalar('Val/acc', acc_val, iter + 1)
        writer.add_scalar('Val/Loss', loss_val.data.item(), iter + 1)
        # ==============save model===============
        torch.save(model, SAVE_PATH + '/KLD_AE_%s_%s.pkl' % (iter + 1, EPOCH))
    writer.close()
