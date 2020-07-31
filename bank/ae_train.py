import os
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd

import parameters as par
from Models import MyAutoencoder
from Custom_Loss_function import MultiLoss_KLD, Loss_fairness_regularization
from Dataset_maker import Dataset_maker
from functions.utils import dataset_splite, plot_loss
from functions.classifier_metrics import accuracy_calculate_gender_ae, EO_evaluation_ae, balanced_accuracy_marital_ae
from functions.classifier_metrics import evaluate

URL = par.BANK_URL
RESULT_SAVE = par.PLOT_RESULT
SAVE_PATH = par.AE_MODEL_SAVE_PATH
TEST_DATA_SAVE = par.CLASSIFIER_TESTSET_PATH
NUM_WORKER = par.NUM_WORKER
FITTING_MODEL = par.FITTING_MODEL

# hyper parameters
EPOCH = par.AE_EPOCH
BATCH_SIZE = par.AE_BATCH_SIZE
LR = par.AE_LR

torch.manual_seed(1)

def model_train(train_loader, val_loader, test_loader, train_val_loader, iteration, learning_rate, save_path, plot_path):
    model = MyAutoencoder()
    ae_loss = MultiLoss_KLD()
    clf_loss = Loss_fairness_regularization()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=learning_rate)

    writer = SummaryWriter(plot_path)
    for iter in range(iteration):
        if iter <= par.C_ITER:
            data4train = train_loader
        else:
            data4train = train_val_loader

        for i, (data, label) in enumerate(train_val_loader):
            data_tensor = Variable(data)
            label_true = Variable(label)
            # ==============forward==============
            data_encoded, data_decoded, pred_label = model(data_tensor)
            loss_AE, _ = ae_loss(data_encoded, data_decoded, data_tensor, label, len(data_tensor))
            loss_CLF, _ = clf_loss(pred_label, label_true)
            loss = loss_AE + loss_CLF
            # ==============backward=============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ==============validation===============
        for j, (data_val, label_val) in enumerate(test_loader):
            data_tensor_val = Variable(data_val)
            label_true_val = Variable(label_val)
            data_encoded_val, data_decoded_val, pred_label_val = model(data_tensor_val)
            loss_AE_val, _ = ae_loss(data_encoded_val, data_decoded_val, data_tensor_val, label_val, len(data_tensor_val))
            loss_CLF_val, _ = clf_loss(pred_label_val, label_true_val)
            loss_val = loss_AE_val + loss_CLF_val
            acc_val, bacc, EO_val, tpr_f, tpr_m, _, _ = evaluate(pred_label_val, label_true_val)
        # =============== log =====================
        writer.add_scalar('Train/Loss', loss.data.item(), iter + 1)
        writer.add_scalar('Test/Loss', loss_val.data.item(), iter + 1)
        writer.add_scalar('Test/EO', EO_val.data.item(), iter + 1)
        writer.add_scalar('Test/acc', acc_val.data.item(), iter + 1)
        writer.add_scalar('Test/bacc', bacc.data.item(), iter + 1)
        writer.add_scalar('Test/TPR female', tpr_f.data.item(), iter + 1)
        writer.add_scalar('Test/TPR male', tpr_m.data.item(), iter + 1)

        # ========================== log =============================
        print('epoch [{}/{}], train loss:{:.4f}, val loss:{:.4f}, val acc:{:.4f}'.format(iter + 1, iteration,
                                                                                         loss.data.item(),
                                                                                         loss_val.data.item(),
                                                                                         acc_val))
        # ==============save model===============
        torch.save(model, save_path + '/KLD_AE_%s_%s.pkl' % (iter + 1, EPOCH))
    writer.close()


def test_save(dataloader, save_path):
    for i, (data, label) in enumerate(dataloader):
        array = np.hstack((data.numpy(), label.numpy()))
    df = pd.DataFrame(array, index=None)
    df.to_csv(save_path, index=False, sep=',')


if __name__ == '__main__':
    # =====make dataset=====
    MyDataset = Dataset_maker(URL, mode='c')
    print(len(MyDataset))
    train_db, val_db, test_db, train_val_db = dataset_splite(MyDataset, 20002, 1000, 20002)

    train_loader = Data.DataLoader(dataset=train_db,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=NUM_WORKER)
    train_val_loader = Data.DataLoader(dataset=train_val_db,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=NUM_WORKER)
    val_loader = Data.DataLoader(dataset=val_db,
                                 batch_size=len(val_db),
                                 shuffle=False,
                                 num_workers=NUM_WORKER)
    test_loader = Data.DataLoader(dataset=test_db,
                                  batch_size=len(test_db),
                                  shuffle=False,
                                  num_workers=NUM_WORKER)
    # ============ save test set ================
    test_save(test_loader, TEST_DATA_SAVE)
    print(TEST_DATA_SAVE)
    # ============= train model =================
    if os.path.exists(SAVE_PATH) is False:
        os.mkdir(SAVE_PATH)
    if os.path.exists(RESULT_SAVE) is False:
        os.mkdir(RESULT_SAVE)

    # training start
    if not par.TESTING:
        model_train(train_loader, val_loader, test_loader, train_val_loader, EPOCH, LR, SAVE_PATH, RESULT_SAVE)
    else:
        model = torch.load(FITTING_MODEL)
        for j, (data_val, label_val) in enumerate(test_loader):
            data_tensor_val = Variable(data_val)
            label_true_val = Variable(label_val)
            data_encoded_val, data_decoded_val, pred_label_val = model(data_tensor_val)
            # balanced_accuracy_gender_ae(test_db, model, mode='evaluate')
            acc_val, bacc, EO_val, tpr_f, tpr_m, tnr_f, tnr_m = evaluate(pred_label_val, label_true_val)

            print (acc_val, bacc, EO_val, tpr_f, tpr_m, tnr_f, tnr_m)