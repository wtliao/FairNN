"""
the multi loss function is made up with three parts:
restruction loss, mse loss and crossEntropy loss
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import parameters as par
import numpy as np
from functions.classifier_metrics import Equalized_odds
from functions.utils import KL_divergence


# ======= autoencoder loss function =========
class MultiLoss_KLD(nn.Module):
    def __init__(self):
        super(MultiLoss_KLD, self).__init__()

    def forward(self, data_encoded, data_decoded, data_true, label_true, batch_size):
        loss_item = []

        # =================== Reconstruction loss ========================
        '''
        For every batch, data_true[0:7] is the numerical data. as well as data_decoded. 
        Calculate mse of [0:7] you need to by 7 with the mse[:,0:7] to get the final mse loss of eight 
        numerical features summation.
        marital: 'married' is tensor[1,0]. 'single' is tensor[0,1]
        '''
        # ======= numerical data loss =========
        numerical_loss = F.mse_loss(data_decoded[:, 0:7], data_true[:, 0:7])
        # ======= norminal data loss ==========
        # job(12)
        job_decoded = data_decoded[:, 7:19]
        job_true = data_true[:, 7:19].nonzero()[:, 1]
        job_loss = F.cross_entropy(job_decoded, job_true)
        # marital (2)
        mari_decoded = data_decoded[:, 19:21]
        mari_true = data_true[:, 19:21].nonzero()[:, 1]
        mari_loss = F.cross_entropy(mari_decoded, mari_true)
        # education (4)
        edu_decoded = data_decoded[:, 21:25]
        edu_true = data_true[:, 21:25].nonzero()[:, 1]
        edu_loss = F.cross_entropy(edu_decoded, edu_true)
        # default (2)
        defa_decoded = data_decoded[:, 25:27]
        defa_true = data_true[:, 25:27].nonzero()[:, 1]
        defa_loss = F.cross_entropy(defa_decoded, defa_true)
        # housing (2)
        hous_decoded = data_decoded[:, 27:29]
        hous_true = data_true[:, 27:29].nonzero()[:, 1]
        hous_loss = F.cross_entropy(hous_decoded, hous_true)
        # loan (2)
        loan_decoded = data_decoded[:, 29:31]
        loan_true = data_true[:, 29:31].nonzero()[:, 1]
        loan_loss = F.cross_entropy(loan_decoded, loan_true)
        # contact (3)
        cont_decoded = data_decoded[:, 31:34]
        cont_true = data_true[:, 31:34].nonzero()[:, 1]
        cont_loss = F.cross_entropy(cont_decoded, cont_true)
        # poutcome (4)
        pout_decoded = data_decoded[:, 34:38]
        pout_true = data_true[:, 34:38].nonzero()[:, 1]
        pout_loss = F.cross_entropy(pout_decoded, pout_true)
        # month (12)
        month_decoded = data_decoded[:, 38:50]
        month_true = data_true[:, 38:50].nonzero()[:, 1]
        month_loss = F.cross_entropy(month_decoded, month_true)

        # ================= KL divergence ====================
        encoded_combined = torch.cat((data_encoded, label_true), 1)
        min_tensor = torch.min(encoded_combined, dim=0)[0]
        max_tensor = torch.max(encoded_combined, dim=0)[0]
        # split batch to married batch and single batch
        single_index = torch.squeeze(torch.nonzero(encoded_combined[:, 11] == 0))
        married_index = torch.squeeze(torch.nonzero(encoded_combined[:, 11] == 1))

        single_batch = torch.index_select(encoded_combined, 0, index=single_index)[:, :-3]
        married_batch = torch.index_select(encoded_combined, 0, index=married_index)[:, :-3]

        single_dis_arr, married_dis_arr = [], []
        for i in range(10):
            single_distr = torch.histc(single_batch[:, i], bins=par.BINS, min=min_tensor[i].item(), max=max_tensor[i].item())\
                           / single_index.shape[0]
            married_distr = torch.histc(married_batch[:, i], bins=par.BINS, min=min_tensor[i].item(), max=max_tensor[i].item()) \
                        / married_index.shape[0]
            single_dis_arr.append(single_distr.detach().numpy())
            married_dis_arr.append(married_distr.detach().numpy())
        # print(single_dis_arr)
        # print(married_dis_arr)

        KLD = KL_divergence(np.array(single_dis_arr), np.array(married_dis_arr))
        mse_Loss = numerical_loss * 7
        ce_Loss = job_loss + mari_loss + edu_loss + defa_loss + hous_loss + loan_loss + cont_loss + pout_loss \
                  + month_loss

        alpha = par.RATIO_KLD
        Multi_loss = (1 - alpha) * (mse_Loss + ce_Loss) + alpha * KLD

        loss_item.append(mse_Loss)
        loss_item.append(ce_Loss)
        loss_item.append(alpha * KLD)

        return Multi_loss, loss_item


# =====================clf loss function=================================
class Loss_fairness_regularization(nn.Module):
    def __init__(self):
        super(Loss_fairness_regularization, self).__init__()

    def forward(self, label_pred, label_true):
        loss_item = []
        # ======binary cross entropy======
        target = label_true[:][:, 0].clone().detach().view(len(label_true), 1)
        CE_loss = F.binary_cross_entropy(label_pred, target)

        # ======equalized odds calculation======
        single_tp, single_tn = 0, 0
        married_tp, married_tn = 0, 0
        single_p, single_n = 0, 0
        married_p, married_n = 0, 0
        married = torch.tensor([1, 0]).type(torch.float32)
        single = torch.tensor([0, 1]).type(torch.float32)
        for i in range(len(label_pred)):
            marital = label_true[i][1:3]
            if torch.equal(marital, married) is True:
                if label_pred[i].detach().numpy() >= par.SIG_THRESHOLD:
                    pred = 1
                    married_p += 1
                    if pred == label_true[i][0].detach().numpy():
                        married_tp += 1
                else:
                    pred = 0
                    married_n += 1
                    if pred == label_true[i][0].detach().numpy():
                        married_tn += 1
            elif torch.equal(marital, single) is True:
                if label_pred[i].detach().numpy() >= par.SIG_THRESHOLD:
                    pred = 1
                    single_p += 1
                    if pred == label_true[i][0].detach().numpy():
                        single_tp += 1
                else:
                    pred = 0
                    single_n += 1
                    if pred == label_true[i][0].detach().numpy():
                        single_tn += 1
        # confusion matrix
        tp = {'married': married_tp, 'single': single_tp}
        tn = {'married': married_tn, 'single': single_tn}
        fp = {'married': married_p - married_tp, 'single': single_p - single_tp}
        fn = {'married': married_n - married_tn, 'single': single_n - single_tn}
        Attr = ['married', 'single']

        # regulization item
        EO_regulization = Equalized_odds(tp, tn, fn, fp, Attr)

        beta = par.RATIO_EO

        Loss_fair = (1 - beta) * CE_loss + beta * EO_regulization
        loss_item.append(CE_loss)
        loss_item.append(EO_regulization)

        return Loss_fair, loss_item

