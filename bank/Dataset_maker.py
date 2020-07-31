import math
import torch
import torch.utils.data as Data
import pandas as pd
from functions.utils import feature_nominal2scalar, numeric_tensor, one_hot_tensor, Upsampling, reweighing_calculate


# ============== original dataset maker ====================
def Dataset_maker(URL, mode=''):
    """
    read data and make the dataset
    """
    # URL = "./data/bank/bank_full_normalized.csv"
    dataset_bank = pd.read_csv(URL)

    """
    data preprocessing
    """
    # ======== map y and month to scalar value =========
    dataset_bank['y'] = dataset_bank['y'].map({'no': 0, 'yes': 1})
    # dataset_bank['month'] = dataset_bank['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    #                                                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})

    # ======== transform nominal attributes to scalar =======
    dataset_bank_test = dataset_bank.copy(deep=True)
    dataset_bank_test['job'] = feature_nominal2scalar(dataset_bank_test['job'])
    dataset_bank_test['marital'] = feature_nominal2scalar(dataset_bank_test['marital'])
    dataset_bank_test['education'] = feature_nominal2scalar(dataset_bank_test['education'])
    dataset_bank_test['default'] = feature_nominal2scalar(dataset_bank_test['default'])
    dataset_bank_test['housing'] = feature_nominal2scalar(dataset_bank_test['housing'])
    dataset_bank_test['month'] = feature_nominal2scalar(dataset_bank_test['month'])
    dataset_bank_test['loan'] = feature_nominal2scalar(dataset_bank_test['loan'])
    dataset_bank_test['contact'] = feature_nominal2scalar(dataset_bank_test['contact'])
    dataset_bank_test['poutcome'] = feature_nominal2scalar(dataset_bank_test['poutcome'])
    # ======== transform numerical attributes to tensor =======
    age_numeric = numeric_tensor(dataset_bank_test['age'])
    balance_numeric = numeric_tensor(dataset_bank_test['balance'])
    day_numeric = numeric_tensor(dataset_bank_test['day'])
    duration_numeric = numeric_tensor(dataset_bank_test['duration'])
    campaign_numeric = numeric_tensor(dataset_bank_test['campaign'])
    pdays_numeric = numeric_tensor(dataset_bank_test['pdays'])
    previous_numeric = numeric_tensor(dataset_bank_test['previous'])
    # ======== transform scalar attributes to one hot code ======
    job_one_hot = one_hot_tensor(dataset_bank_test['job'])
    mari_one_hot = one_hot_tensor(dataset_bank_test['marital'])
    edu_one_hot = one_hot_tensor(dataset_bank_test['education'])
    defa_one_hot = one_hot_tensor(dataset_bank_test['default'])
    hous_one_hot = one_hot_tensor(dataset_bank_test['housing'])
    loan_one_hot = one_hot_tensor(dataset_bank_test['loan'])
    cont_one_hot = one_hot_tensor(dataset_bank_test['contact'])
    poutcome_one_hot = one_hot_tensor(dataset_bank_test['poutcome'])
    month_one_hot = one_hot_tensor(dataset_bank_test['month'])

    y_label = numeric_tensor(dataset_bank_test['y'])

    # ======== combine all attributes to dataset ========
    dataset_bank_tensor = torch.cat((age_numeric, balance_numeric, day_numeric, duration_numeric, campaign_numeric,
                                     pdays_numeric, previous_numeric, job_one_hot, mari_one_hot, edu_one_hot,
                                     defa_one_hot, hous_one_hot, loan_one_hot, cont_one_hot, poutcome_one_hot,
                                     month_one_hot), 1)

    if mode == '':
        Mydataset = Data.TensorDataset(dataset_bank_tensor, y_label)
    elif mode == 'c':
        # construct a combine-label
        combine_label = torch.cat((y_label, mari_one_hot), 1)
        Mydataset = Data.TensorDataset(dataset_bank_tensor, combine_label)

    return Mydataset

# ================== classification dataset maker ====================
def Dataset_maker_classification(URL):
    encoded_bank_dataset_test = pd.read_csv(URL)
    print(encoded_bank_dataset_test.shape)
    encoded_bank_dataset_test['y'] = encoded_bank_dataset_test['y'].map({'no': 0, 'yes': 1})
    encoded_bank_dataset_test['marital'] = feature_nominal2scalar(encoded_bank_dataset_test['marital'])
    marital_one_hot = one_hot_tensor(encoded_bank_dataset_test['marital'])
    # ======transform numerical attribute to tensor ======
    f1_numeric = numeric_tensor(encoded_bank_dataset_test['feature_1'])
    f2_numeric = numeric_tensor(encoded_bank_dataset_test['feature_2'])
    f3_numeric = numeric_tensor(encoded_bank_dataset_test['feature_3'])
    f4_numeric = numeric_tensor(encoded_bank_dataset_test['feature_4'])
    f5_numeric = numeric_tensor(encoded_bank_dataset_test['feature_5'])
    f6_numeric = numeric_tensor(encoded_bank_dataset_test['feature_6'])
    f7_numeric = numeric_tensor(encoded_bank_dataset_test['feature_7'])
    f8_numeric = numeric_tensor(encoded_bank_dataset_test['feature_8'])
    f9_numeric = numeric_tensor(encoded_bank_dataset_test['feature_9'])
    f10_numeric = numeric_tensor(encoded_bank_dataset_test['feature_10'])
    y_label = numeric_tensor(encoded_bank_dataset_test['y'])
    # =======combine all attributes to dataset =======
    dataset_encoded = torch.cat((f1_numeric, f2_numeric, f3_numeric, f4_numeric, f5_numeric, f6_numeric, f7_numeric,
                                 f8_numeric, f9_numeric, f10_numeric), 1)

    # construct a combine-label
    combine_label = torch.cat((y_label, marital_one_hot), 1)
    Mydataset = Data.TensorDataset(dataset_encoded, combine_label)

    return Mydataset


def Dataset_maker_clf_distribution(URL, proportion):
    """
    data: dataset,
    proportion (list): [train, val, test]
    """
    quantity = []
    dataset_list = []
    dataset_tensor_list = []
    married_set_num, single_set_num = [], []

    data_test = pd.read_csv(URL)
    # separate dataset in two status
    data_married = data_test.loc[data_test['marital'] == 'married']
    data_single = data_test.loc[data_test['marital'] == 'single']
    data_married.reset_index(drop=True, inplace=True)
    data_single.reset_index(drop=True, inplace=True)
    # regenerate train, val, test with equal gender distribution
    quantity.append(len(data_married))
    quantity.append(len(data_single))
    for i in range(2):
        train_num = quantity[i] - math.ceil(quantity[i] * proportion[1]) - math.ceil(quantity[i] * proportion[2])
        val_num = math.ceil(quantity[i] * proportion[1])
        test_num = math.ceil(quantity[i] * proportion[2])
        if i == 0:
            married_set_num.append(train_num)
            married_set_num.append(val_num)
            married_set_num.append(test_num)
        elif i == 1:
            single_set_num.append(train_num)
            single_set_num.append(val_num)
            single_set_num.append(test_num)

    # step 2: shuffle married and single dataset respectively
    data_married = data_married.sample(frac=1.0)
    data_married.reset_index(drop=True, inplace=True)
    train_married = data_married.loc[0: married_set_num[0] - 1]
    val_married = data_married.loc[married_set_num[0]: sum(married_set_num[0:2]) - 1]
    test_married = data_married.loc[sum(married_set_num[0:2]): sum(married_set_num)]

    data_single = data_single.sample(frac=1.0)
    data_single.reset_index(drop=True, inplace=True)
    train_single = data_single.loc[0: single_set_num[0] - 1]
    val_single = data_single.loc[single_set_num[0]: sum(single_set_num[0:2]) - 1]
    test_single = data_single.loc[sum(single_set_num[0:2]): sum(single_set_num)]

    # step 3: connect married and single data set together
    train_data = pd.concat([train_married, train_single])
    val_data = pd.concat([val_married, val_single])
    test_data = pd.concat([test_married, test_single])

    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    # step 4: shuffle val dataset and test dataset
    val_head = val_data.loc[0:0]
    val_shuffle = val_data.loc[1:len(val_data)].sample(frac=1.0)
    val_data = pd.concat([val_head, val_shuffle])
    val_data.reset_index(drop=True, inplace=True)
    test_head = test_data.loc[0:0]
    test_shuffle = test_data.loc[1:len(test_data)].sample(frac=1.0)
    test_data = pd.concat([test_head, test_shuffle])
    test_data.reset_index(drop=True, inplace=True)

    dataset_list.append(train_data)
    dataset_list.append(val_data)
    dataset_list.append(test_data)

    # transform dataframe to tensor
    for j in range(2):
        dataset_list[j]['y'] = dataset_list[j]['y'].map({'no': 0, 'yes': 1})
        dataset_list[j]['marital'] = feature_nominal2scalar(dataset_list[j]['marital'])
        marital_one_hot = one_hot_tensor(dataset_list[j]['marital'])

        # ======transform numerical attribute to tensor ======
        f1_numeric = numeric_tensor(dataset_list[j]['feature_1'])
        f2_numeric = numeric_tensor(dataset_list[j]['feature_2'])
        f3_numeric = numeric_tensor(dataset_list[j]['feature_3'])
        f4_numeric = numeric_tensor(dataset_list[j]['feature_4'])
        f5_numeric = numeric_tensor(dataset_list[j]['feature_5'])
        f6_numeric = numeric_tensor(dataset_list[j]['feature_6'])
        f7_numeric = numeric_tensor(dataset_list[j]['feature_7'])
        f8_numeric = numeric_tensor(dataset_list[j]['feature_8'])
        f9_numeric = numeric_tensor(dataset_list[j]['feature_9'])
        f10_numeric = numeric_tensor(dataset_list[j]['feature_10'])
        y_label = numeric_tensor(dataset_list[j]['y'])

        dataset_encoded = torch.cat((f1_numeric, f2_numeric, f3_numeric, f4_numeric, f5_numeric, f6_numeric, f7_numeric,
                                     f8_numeric, f9_numeric, f10_numeric), 1)
        # construct a combine-label
        combine_label = torch.cat((y_label, marital_one_hot), 1)
        Mydataset = Data.TensorDataset(dataset_encoded, combine_label)

        dataset_tensor_list.append(Mydataset)

    return dataset_tensor_list[0], dataset_tensor_list[1], test_data

