import torch
import os
import pandas as pd
import torch.utils.data as Data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import parameters as par

from Dataset_maker import Dataset_maker
from functions.utils import dataset_splite

AE_MODEL = par.FITTING_MODEL
URL = par.CLASSIFIER_TESTSET_PATH


def read_testset(path):
    array = pd.read_csv(path).to_numpy()
    tensor_test = torch.from_numpy(array).type(torch.FloatTensor)
    testDataset = Data.TensorDataset(tensor_test[:, 0:50], tensor_test[:, 50:])
    return testDataset


def encoded_visualization(model, dataset_loader, save_path, mode):
    for num, (data, label) in enumerate(dataset_loader):
        data_tensor = data
        data_tensor = Variable(data_tensor)
        data_encoded, _, _ = model(data_tensor)
        data_label = label
    if mode == '2clusters':
        # double kinds of labels
        label_list = []
        meta_header = ['Marital\tY']

        for i, data_l in enumerate(data_label):
            if torch.equal(data_l[1:3].type(torch.int64), torch.tensor([1, 0])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(0)) is True:
                gender = 'Married'
                income = 'no'
                label = [str(gender) + '\t' + str(income)]
                label_list.append(label)
            elif torch.equal(data_l[1:3].type(torch.int64), torch.tensor([0, 1])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(0)) is True:
                gender = 'Single'
                income = 'no'
                label = [str(gender) + '\t' + str(income)]
                label_list.append(label)
            elif torch.equal(data_l[1:3].type(torch.int64), torch.tensor([1, 0])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(1)) is True:
                gender = 'Married'
                income = 'yes'
                label = [str(gender) + '\t' + str(income)]
                label_list.append(label)
            elif torch.equal(data_l[1:3].type(torch.int64), torch.tensor([0, 1])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(1)) is True:
                gender = 'Single'
                income = 'yes'
                label = [str(gender) + '\t' + str(income)]
                label_list.append(label)

        writer_test = SummaryWriter(logdir=save_path)
        print(label_list)
        # writer_test.add_embedding(data_encoded, metadata=data_label)
        writer_test.add_embedding(data_encoded, metadata=label_list, metadata_header=meta_header)
        writer_test.close()

    elif mode == '4clusters':
        # four clusters with different colors
        label_list = []
        # meta_header = ['Gender\tIncome']

        for i, data_l in enumerate(data_label):
            if torch.equal(data_l[1:3].type(torch.int64), torch.tensor([1, 0])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(0)) is True:
                gender = 'Married'
                income = 'no'
                label = [str(gender) + str(income)]
                label_list.append(label)
            elif torch.equal(data_l[1:3].type(torch.int64), torch.tensor([0, 1])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(0)) is True:
                gender = 'Single'
                income = 'no'
                # label = [str(gender) + '\t' + str(income)]
                label = [str(gender) + str(income)]
                label_list.append(label)
            elif torch.equal(data_l[1:3].type(torch.int64), torch.tensor([1, 0])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(1)) is True:
                gender = 'Married'
                income = 'yes'
                label = [str(gender) + str(income)]
                label_list.append(label)
            elif torch.equal(data_l[1:3].type(torch.int64), torch.tensor([0, 1])) is True and \
                    torch.equal(data_l[0].type(torch.int64), torch.tensor(1)) is True:
                gender = 'Single'
                income = 'yes'
                label = [str(gender) + str(income)]
                label_list.append(label)

        writer_test = SummaryWriter(logdir=save_path)
        writer_test.add_embedding(data_encoded, metadata=label_list)
        # writer_test.add_embedding(data_encoded, metadata=label_list, metadata_header=meta_header)
        writer_test.close()


# ================== visualization =====================
if __name__ == '__main__':
    VISUAL_SAVE = par.VISUAL_SAVE + '/AE_KLD_0_%s_EO_0_%s' % (int(par.RATIO_KLD*10), int(par.RATIO_EO*10))
    print('KLD:%s, EO:%s' % (par.RATIO_KLD, par.RATIO_EO))
    print('visual path:', VISUAL_SAVE)
    if os.path.exists(VISUAL_SAVE) is False:
        os.mkdir(VISUAL_SAVE)
    model = torch.load(AE_MODEL)
    test_db = read_testset(URL)
    test_loader = Data.DataLoader(dataset=test_db, batch_size=len(test_db), shuffle=False, num_workers=0)
    encoded_visualization(model, test_loader, VISUAL_SAVE, mode='2clusters')
