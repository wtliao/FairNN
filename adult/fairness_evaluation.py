import torch
import argparse
import pandas as pd
import torch.utils.data as Data
import parameters as par
from Dataset_maker import Dataset_maker_classification
from functions.classifier_metrics import balanced_accuracy_gender_ae


# coefficient outsider
parser = argparse.ArgumentParser()
parser.add_argument('--m', dest='mode', help='t, p', default='t', type=str)
args = parser.parse_args()

if args.mode == 't':
    URL = par.CLASSIFIER_TESTSET_PATH
    FITTING_MODEL = par.FITTING_MODEL
elif args.mode == 'p':
    URL = par.P_TEST_PATH
    FITTING_MODEL = par.P_FITTING_MODEL

# hyper parameter
print('JSD:%s,EO:%s' % (par.RATIO_KLD, par.RATIO_EO))
URL = par.CLASSIFIER_TESTSET_PATH
URL_P = par.P_TEST_PATH
FITTING_MODEL = par.FITTING_MODEL
FITTING_MODEL_P = par.P_FITTING_MODEL

def read_testset(path):
    array = pd.read_csv(path).to_numpy()
    tensor_test = torch.from_numpy(array).type(torch.FloatTensor)
    testDataset = Data.TensorDataset(tensor_test[:, 0:99], tensor_test[:, 99:102])
    return testDataset


def model_test(dataloader, model):
    dataset = dataloader
    model_test = model
    balanced_accuracy_gender_ae(dataset, model_test, mode='evaluate')


if __name__ == '__main__':
    model = torch.load(FITTING_MODEL)
    print(FITTING_MODEL)
    print(URL)
    test_db = read_testset(URL)
    test_loader = Data.DataLoader(dataset=test_db, batch_size=len(test_db), shuffle=False, num_workers=0)
    model_test(test_db, model)
