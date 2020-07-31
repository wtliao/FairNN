import torch
import pandas as pd
import torch.utils.data as Data
from torch.autograd import Variable
import parameters as par

from Dataset_maker import Dataset_maker

MODEL_PATH = par.FITTING_MODEL
URL = par.BANK_URL
SAVE_PATH = par.ENCODED_DATA
print("Bank fitting iteration:", par.FITTING_ITER)

MyDataset = Dataset_maker(URL)
model = torch.load(MODEL_PATH)

dataset_loader = Data.DataLoader(dataset=MyDataset,
                                 batch_size=len(MyDataset),
                                 shuffle=False,
                                 num_workers=0)

for num, (data, _) in enumerate(dataset_loader):
    data_tensor = data
    data_tensor = Variable(data_tensor)
    data_encoded, _ = model(data_tensor)

# transfor variable to dataframe
encoded_data = pd.DataFrame(data_encoded.data.numpy(), columns=['feature_1', 'feature_2', 'feature_3', 'feature_4',
                                                                'feature_5', 'feature_6', 'feature_7', 'feature_8',
                                                                'feature_9', 'feature_10'])

extra_data = pd.read_csv(URL)[['marital', 'y']]

encoded_dataset = pd.concat([encoded_data, extra_data], axis=1)
print(encoded_dataset.shape)
encoded_dataset.to_csv(SAVE_PATH, index=False, sep=',')
print('bank encoded dataset generated')