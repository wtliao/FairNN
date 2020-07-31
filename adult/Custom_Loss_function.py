"""
the multi loss function is made up with three parts:
restruction loss, mse loss and crossEntropy loss
"""
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import parameters as par
import numpy as np
from functions.classifier_metrics import Equalized_odds
from functions.utils import KL_divergence


# ======= autoencoder loss function =========
class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()

    def decoded_crossentropy(self, data_decoded, start_index, end_index):
        return torch.index_select(data_decoded, 1,
                                  torch.linspace(start_index, end_index, (end_index - start_index + 1)).long())

    def true_crossentropy(self, data_true, start_index, end_index, batch_size):
        data_true_index = torch.ones(batch_size, 1)
        data_true_slice = torch.index_select(data_true, 1, torch.linspace(start_index, end_index,
                                                                          (end_index - start_index + 1)).long())
        for i in range(batch_size):
            data_true_index[i] = data_true_slice[i].nonzero()[0]
        return data_true_index.long().squeeze(1)

    def forward(self, data_decoded, data_true, batch_size):
        loss_item = []

        # Reconstruction_loss = F.mse_loss(data_decoded, data_true)
        # the following are separated feature loss
        # ===age loss===
        age_loss = F.mse_loss(data_decoded[:][:, 0], data_true[:][:, 0])
        # ===workclass loss===
        work_decoded = data_decoded[:, 1:10]  # self.decoded_crossentropy(data_decoded, 1, 9)
        work_true = data_true[:, 1:10].nonzero()[:, 1]  # self.true_crossentropy(data_true, 1, 9, batch_size)
        work_loss = F.cross_entropy(work_decoded, work_true)
        # ===edu loss===
        edu_decoded = data_decoded[:, 10:26]  # self.decoded_crossentropy(data_decoded, 10, 25)
        edu_true = data_true[:, 10:26].nonzero()[:, 1]  # self.true_crossentropy(data_true, 10, 25, batch_size)
        edu_loss = F.cross_entropy(edu_decoded, edu_true)
        # ===edu_num loss===
        edu_num_loss = F.mse_loss(data_decoded[:][:, 26], data_true[:][:, 26])
        # ===marital loss===
        marital_decoded = data_decoded[:, 27:34]  # self.decoded_crossentropy(data_decoded, 27, 33)
        marital_true = data_true[:, 27:34].nonzero()[:, 1]  # self.true_crossentropy(data_true, 27, 33, batch_size)
        marital_loss = F.cross_entropy(marital_decoded, marital_true)
        # ===occupation loss===
        occupation_decoded = data_decoded[:, 34:49]  # self.decoded_crossentropy(data_decoded, 34, 48)
        occupation_true = data_true[:, 34:49].nonzero()[:, 1]  # self.true_crossentropy(data_true, 34, 48, batch_size)
        occupation_loss = F.cross_entropy(occupation_decoded, occupation_true)
        # ===relationship loss===
        relationship_decoded = data_decoded[:, 49:55]  # self.decoded_crossentropy(data_decoded, 49, 54)
        relationship_true = data_true[:, 49:55].nonzero()[:, 1]  # self.true_crossentropy(data_true, 49, 54, batch_size)
        relationship_loss = F.cross_entropy(relationship_decoded, relationship_true)
        # ===race loss===
        race_decoded = data_decoded[:, 55:60]  # self.decoded_crossentropy(data_decoded, 55, 59)
        race_true = data_true[:, 55:60].nonzero()[:, 1]  # self.true_crossentropy(data_true, 55, 59, batch_size)
        race_loss = F.cross_entropy(race_decoded, race_true)
        # ===sex loss===
        sex_decoded = data_decoded[:, 60:62]  # self.decoded_crossentropy(data_decoded, 60, 61)
        sex_true = data_true[:, 60:62].nonzero()[:, 1]  # self.true_crossentropy(data_true, 60, 61, batch_size)
        sex_loss = F.cross_entropy(sex_decoded, sex_true)
        # ===capital gain loss===
        capgain_loss = F.mse_loss(data_decoded[:][:, 62], data_true[:][:, 62])
        # ===capital loss loss===
        caploss_loss = F.mse_loss(data_decoded[:][:, 63], data_true[:][:, 63])
        # ===hour.per.week loss===
        hour_loss = F.mse_loss(data_decoded[:][:, 64], data_true[:][:, 64])
        # ===native loss===
        native_decoded = data_decoded[:, 65:107]  # self.decoded_crossentropy(data_decoded, 65, 106)
        native_true = data_true[:, 65:107].nonzero()[:, 1]  # self.true_crossentropy(data_true, 65, 106, batch_size)
        native_loss = F.cross_entropy(native_decoded, native_true)

        Multi_loss = age_loss + work_loss + edu_loss + edu_num_loss + marital_loss + occupation_loss + relationship_loss + race_loss + sex_loss + capgain_loss + caploss_loss + hour_loss + native_loss
        mse_Loss = age_loss + edu_num_loss + capgain_loss + caploss_loss + hour_loss
        ce_Loss = work_loss + edu_loss + marital_loss + occupation_loss + relationship_loss + race_loss + sex_loss+ native_loss
        loss_item.append(mse_Loss)
        loss_item.append(ce_Loss)

        return Multi_loss, loss_item


class MultiLoss_KLD(nn.Module):
    def __init__(self):
        super(MultiLoss_KLD, self).__init__()

    # def decoded_crossentropy(self, data_decoded, start_index, end_index):
    #     return torch.index_select(data_decoded, 1,
    #                               torch.linspace(start_index, end_index, (end_index - start_index + 1)).long())
    #
    # def true_crossentropy(self, data_true, start_index, end_index, batch_size):
    #     data_true_index = torch.ones(batch_size, 1)
    #     data_true_slice = torch.index_select(data_true, 1, torch.linspace(start_index, end_index,
    #                                                                       (end_index - start_index + 1)).long())
    #     for i in range(batch_size):
    #         data_true_index[i] = data_true_slice[i].nonzero()[0]
    #     return data_true_index.long().squeeze(1)

    def forward(self, data_encoded, data_decoded, data_true, label_true, batch_size):
        loss_item = []

        # =================== Reconstruction loss ========================
        # ===age loss===
        age_loss = F.mse_loss(data_decoded[:][:, 0], data_true[:][:, 0])
        # ===workclass loss===
        work_decoded = data_decoded[:, 1:8]
        work_true = data_true[:, 1:8].nonzero()[:, 1]
        work_loss = F.cross_entropy(work_decoded, work_true)
        # ===edu loss===
        edu_decoded = data_decoded[:, 8:24]
        edu_true = data_true[:, 8:24].nonzero()[:, 1]
        edu_loss = F.cross_entropy(edu_decoded, edu_true)
        # ===marital loss===
        marital_decoded = data_decoded[:, 24:31]
        marital_true = data_true[:, 24:31].nonzero()[:, 1]
        marital_loss = F.cross_entropy(marital_decoded, marital_true)
        # ===occupation loss===
        occupation_decoded = data_decoded[:, 31:45]
        occupation_true = data_true[:, 31:45].nonzero()[:, 1]
        occupation_loss = F.cross_entropy(occupation_decoded, occupation_true)
        # ===relationship loss===
        relationship_decoded = data_decoded[:, 45:51]
        relationship_true = data_true[:, 45:51].nonzero()[:, 1]
        relationship_loss = F.cross_entropy(relationship_decoded, relationship_true)
        # ===race loss===
        race_decoded = data_decoded[:, 51:53]
        race_true = data_true[:, 51:53].nonzero()[:, 1]
        race_loss = F.cross_entropy(race_decoded, race_true)
        # ===sex loss===
        sex_decoded = data_decoded[:, 53:55]
        sex_true = data_true[:, 53:55].nonzero()[:, 1]
        sex_loss = F.cross_entropy(sex_decoded, sex_true)
        # ===capital gain loss===
        capgain_loss = F.mse_loss(data_decoded[:][:, 55], data_true[:][:, 55])
        # ===capital loss loss===
        caploss_loss = F.mse_loss(data_decoded[:][:, 56], data_true[:][:, 56])
        # ===hour.per.week loss===
        hour_loss = F.mse_loss(data_decoded[:][:, 57], data_true[:][:, 57])
        # ===native loss===
        native_decoded = data_decoded[:, 58:99]
        native_true = data_true[:, 58:99].nonzero()[:, 1]
        native_loss = F.cross_entropy(native_decoded, native_true)

        # ================= KL divergence ====================
        encoded_combined = torch.cat((data_encoded, label_true), 1)
        min_tensor = torch.min(encoded_combined, dim=0)[0]
        max_tensor = torch.max(encoded_combined, dim=0)[0]
        # split batch to male batch and female batch
        male_index = torch.squeeze(torch.nonzero(encoded_combined[:, 11] == 0))
        female_index = torch.squeeze(torch.nonzero(encoded_combined[:, 11] == 1))

        male_batch = torch.index_select(encoded_combined, 0, index=male_index)[:, :-3]
        female_batch = torch.index_select(encoded_combined, 0, index=female_index)[:, :-3]
        '''
        second kl divergence calculation approach on multiple gaussian distribution
        '''
        # find distribution of male and female batch
        male_dis_arr, female_dis_arr = [], []
        for i in range(10):
            male_distr = torch.histc(male_batch[:, i], bins=par.KLD_BINS, min=min_tensor[i].item(), max=max_tensor[i].item()) \
                        / male_index.shape[0]
            female_distr = torch.histc(female_batch[:, i], bins=par.KLD_BINS, min=min_tensor[i].item(), max=max_tensor[i].item()) \
                        / female_index.shape[0]
            male_dis_arr.append(male_distr.detach().numpy())
            female_dis_arr.append(female_distr.detach().numpy())
            # KL divergence calculation
        KLD = KL_divergence(np.array(male_dis_arr), np.array(female_dis_arr))

        mse_Loss = age_loss + capgain_loss + caploss_loss + hour_loss
        ce_Loss = work_loss + edu_loss + marital_loss + occupation_loss + relationship_loss \
                  + race_loss + sex_loss + native_loss
        # different alpha 0.1 ~ 1
        alpha = par.RATIO_KLD
        Multi_loss = (1 - alpha) * (mse_Loss + ce_Loss) + alpha * KLD

        loss_item.append(mse_Loss)
        loss_item.append(ce_Loss)
        loss_item.append(alpha * KLD)

        return Multi_loss, loss_item


# ======== classification network loss function ========
class ReweighLoss(nn.Module):
    def __init__(self):
        super(ReweighLoss, self).__init__()

    def forward(self, label_pred, label_true, reweigh_matrix):
        count_list = [0, 0, 0, 0]
        male_less = torch.tensor([0, 0, 1]).type(torch.float32)
        male_more = torch.tensor([1, 0, 1]).type(torch.float32)
        female_less = torch.tensor([0, 1, 0]).type(torch.float32)
        female_more = torch.tensor([1, 1, 0]).type(torch.float32)

        for i in range(len(label_true)):
            if torch.equal(label_true[i], male_less) is True:
                if count_list[0] == 0:
                    true_m_l = label_true[i]
                    pred_m_l = label_pred[i]
                elif count_list[0] == 1:
                    true_m_l = torch.stack((true_m_l, label_true[i]), 0)
                    pred_m_l = torch.stack((pred_m_l, label_pred[i]), 0)
                else:
                    true_m_l = torch.cat((true_m_l, label_true[i].view(1, 3)), 0)
                    pred_m_l = torch.cat((pred_m_l, label_pred[i].view(1, 1)), 0)
                count_list[0] += 1
            elif torch.equal(label_true[i], male_more) is True:
                if count_list[1] == 0:
                    true_m_m = label_true[i]
                    pred_m_m = label_pred[i]
                elif count_list[1] == 1:
                    true_m_m = torch.stack((true_m_m, label_true[i]), 0)
                    pred_m_m = torch.stack((pred_m_m, label_pred[i]), 0)
                else:
                    true_m_m = torch.cat((true_m_m, label_true[i].view(1, 3)), 0)
                    pred_m_m = torch.cat((pred_m_m, label_pred[i].view(1, 1)), 0)
                count_list[1] += 1
            elif torch.equal(label_true[i], female_less) is True:
                if count_list[2] == 0:
                    true_f_l = label_true[i]
                    pred_f_l = label_pred[i]
                elif count_list[2] == 1:
                    true_f_l = torch.stack((true_f_l, label_true[i]), 0)
                    pred_f_l = torch.stack((pred_f_l, label_pred[i]), 0)
                else:
                    true_f_l = torch.cat((true_f_l, label_true[i].view(1, 3)), 0)
                    pred_f_l = torch.cat((pred_f_l, label_pred[i].view(1, 1)), 0)
                count_list[2] += 1
            elif torch.equal(label_true[i], female_more) is True:
                if count_list[3] == 0:
                    true_f_m = label_true[i]
                    pred_f_m = label_pred[i]
                elif count_list[3] == 1:
                    true_f_m = torch.stack((true_f_m, label_true[i]), 0)
                    pred_f_m = torch.stack((pred_f_m, label_pred[i]), 0)
                else:
                    true_f_m = torch.cat((true_f_m, label_true[i].view(1, 3)), 0)
                    pred_f_m = torch.cat((pred_f_m, label_pred[i].view(1, 1)), 0)
                count_list[3] += 1

        # ====== bce calculate ========
        loss_m_l = F.binary_cross_entropy(pred_m_l, true_m_l[:][:, 0].clone().detach().view(len(true_m_l), 1))
        loss_m_m = F.binary_cross_entropy(pred_m_m, true_m_m[:][:, 0].clone().detach().view(len(true_m_m), 1))
        loss_f_l = F.binary_cross_entropy(pred_f_l, true_f_l[:][:, 0].clone().detach().view(len(true_f_l), 1))
        loss_f_m = F.binary_cross_entropy(pred_f_m, true_f_m[:][:, 0].clone().detach().view(len(true_f_m), 1))
        # ====== reweigh loss ========
        loss = (reweigh_matrix.iloc[0, 0] * loss_m_l +
                reweigh_matrix.iloc[0, 1] * loss_m_m +
                reweigh_matrix.iloc[1, 0] * loss_f_l +
                reweigh_matrix.iloc[1, 1] * loss_f_m) / 4

        return loss


# =============================== clf loss function ==============================================
class Loss_fairness_regularization(nn.Module):
    def __init__(self):
        super(Loss_fairness_regularization, self).__init__()

    def forward(self, label_pred, label_true):
        loss_item = []
        # ======binary cross entropy======
        target = label_true[:][:, 0].clone().detach().view(len(label_true), 1)
        CE_loss = F.binary_cross_entropy(label_pred, target)

        # ======equalized odds calculation======
        male_tp, male_tn = 0, 0
        female_tp, female_tn = 0, 0
        male_p, male_n = 0, 0
        female_p, female_n = 0, 0
        female = torch.tensor([1, 0]).type(torch.float32)
        male = torch.tensor([0, 1]).type(torch.float32)
        for i in range(len(label_pred)):
            gender = label_true[i][1:3]
            if torch.equal(gender, female) is True:
                if label_pred[i].detach().numpy() >= par.SIG_THRESHOLD:
                    pred = 1
                    female_p += 1
                    if pred == label_true[i][0].detach().numpy():
                        female_tp += 1
                else:
                    pred = 0
                    female_n += 1
                    if pred == label_true[i][0].detach().numpy():
                        female_tn += 1
            elif torch.equal(gender, male) is True:
                if label_pred[i].detach().numpy() >= par.SIG_THRESHOLD:
                    pred = 1
                    male_p += 1
                    if pred == label_true[i][0].detach().numpy():
                        male_tp += 1
                else:
                    pred = 0
                    male_n += 1
                    if pred == label_true[i][0].detach().numpy():
                        male_tn += 1
        # confusion matrix
        tp = {'female': female_tp, 'male': male_tp}
        tn = {'female': female_tn, 'male': male_tn}
        fp = {'female': female_p - female_tp, 'male': male_p - male_tp}
        fn = {'female': female_n - female_tn, 'male': male_n - male_tn}
        Attr = ['female', 'male']

        # regulization item
        EO_regulization = Equalized_odds(tp, tn, fn, fp, Attr)

        beta = par.RATIO_EO

        Loss_fair = (1 - beta) * CE_loss + beta * EO_regulization
        loss_item.append(CE_loss)
        loss_item.append(EO_regulization)

        return Loss_fair, loss_item


class Loss_function_P_sampling(nn.Module):
    def __init__(self):
        super(Loss_function_P_sampling, self).__init__()

    def forward(self, label_pred, label_true, data):
        target = label_true[:][:, 0].clone().detach().view(len(label_true), 1)
        CE_loss = F.binary_cross_entropy(label_pred, target)

        # ======= preferential sampling =========
        expected_amount, actual_amount = [], []
        sex_list, income_list = [], []
        dataset = torch.cat((data, label_true, label_pred), 1)
        index_DP = torch.squeeze(torch.nonzero((dataset[:, 108] == 1) + (dataset[:, 107] == 1) == 2))
        index_PP = torch.squeeze(torch.nonzero((dataset[:, 108] == 0) + (dataset[:, 107] == 1) == 2))
        index_DN = torch.squeeze(torch.nonzero((dataset[:, 108] == 1) + (dataset[:, 107] == 0) == 2))
        index_PN = torch.squeeze(torch.nonzero((dataset[:, 108] == 0) + (dataset[:, 107] == 0) == 2))

        DP_batch = torch.index_select(dataset, 0, index=index_DP)
        PP_batch = torch.index_select(dataset, 0, index=index_PP)
        DN_batch = torch.index_select(dataset, 0, index=index_DN)
        PN_batch = torch.index_select(dataset, 0, index=index_PN)
        # sort
        sex_list.append(len(DP_batch) + len(DN_batch))
        sex_list.append(len(PP_batch) + len(PN_batch))
        income_list.append(len(DP_batch) + len(PP_batch))
        income_list.append(len(DN_batch) + len(PN_batch))
        # DP, DN, PP, PN
        for i in range(2):
            for j in range(2):
                expected_amount.append(math.ceil(sex_list[i] * income_list[j] / len(data)))
        DP_batch_new = self.Duplicating(DP_batch, expected_amount[0], name='DP')
        DN_batch_new = self.Skipping(DN_batch, expected_amount[1], name='DN')
        PP_batch_new = self.Skipping(PP_batch, expected_amount[2], name='PP')
        PN_batch_new = self.Duplicating(PN_batch, expected_amount[3], name='PN')
        new_train_set = torch.cat((DP_batch_new, DN_batch_new, PP_batch_new, PN_batch_new), 0)[:, 0:110]

        return CE_loss, new_train_set

    def Duplicating(self, batch, expected_value, name=''):
        if name == 'DP':
            batch_sort = torch.index_select(batch, 0, index=batch[:, 110].sort()[1])
        elif name == 'PN':
            batch_sort = torch.index_select(batch, 0, index=batch[:, 110].sort(descending=True)[1])
        else:
            raise Exception("choose one community 'PN'or 'DP'!!")

        need_value = expected_value - len(batch_sort)

        if need_value < len(batch_sort):
            duplicate = batch_sort[0:need_value, :]
            batch_new = torch.cat((batch_sort, duplicate), 0)
        elif need_value > len(batch_sort):
            circle = math.floor(need_value / len(batch_sort))
            duplicate = batch_sort
            for i in range(circle):
                duplicate = torch.cat((duplicate, batch_sort), 0)
            rest = batch_sort[0:(need_value - circle * len(batch_sort)), :]
            batch_new = torch.cat((rest, duplicate), 0)
        return batch_new

    def Skipping(self, batch, expected_value, name=''):
        if name == 'PP':
            batch_sort = torch.index_select(batch, 0, index=batch[:, 110].sort()[1])
        elif name == 'DN':
            batch_sort = torch.index_select(batch, 0, index=batch[:, 110].sort(descending=True)[1])
            print(len(batch_sort))
            print(expected_value)
        else:
            raise Exception("choose one community 'PP'or 'DN'")

        drop_value = len(batch_sort) - expected_value

        if drop_value < 0:
            raise Exception('skipping value too large!!')
        elif drop_value > 0 and drop_value > len(batch_sort):
            raise Exception('expected value is illegal negative!!')
        else:
            batch_rest = batch_sort[drop_value:, :]
        return batch_rest
