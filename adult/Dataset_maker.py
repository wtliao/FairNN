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
    # URL = "./data/adult_census_income/adult_normalized.csv"
    dataset_adult = pd.read_csv(URL)
    data_test = dataset_adult[['age', 'workclass', 'education', 'marital.status', 'occupation',
                               'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week',
                               'native.country', 'income']]
    """
    data preprocessing
    """
    # ====== outlier detection and remove ========
    # outlier_list = (data_test.loc[(data_test['sex'] == 'Female') & (
    #         data_test['relationship'] == 'Husband')].index.values).tolist()
    # outlier_list.extend((data_test.loc[(data_test['sex'] == 'Male') & (
    #             data_test['relationship'] == 'Wife')].index.values).tolist())
    # for i, index in enumerate(outlier_list):
    #     data_test.drop([index], inplace=True)
    # data_test = data_test.reset_index(drop=True)
    # ======= map income to scalar value ========
    # data_test['income'] = data_test['income'].map({'<=50K': 0, '>50K': 1})

    # =======transform nominal attributes to scalar==========
    dataset_test_copy = data_test.copy(deep=True)
    dataset_test_copy['education'] = feature_nominal2scalar(data_test['education'])
    dataset_test_copy['workclass'] = feature_nominal2scalar(data_test['workclass'])
    dataset_test_copy['marital.status'] = feature_nominal2scalar(data_test['marital.status'])
    dataset_test_copy['occupation'] = feature_nominal2scalar(data_test['occupation'])
    dataset_test_copy['relationship'] = feature_nominal2scalar(data_test['relationship'])
    dataset_test_copy['race'] = feature_nominal2scalar(data_test['race'])
    dataset_test_copy['sex'] = feature_nominal2scalar(data_test['sex'])
    dataset_test_copy['native.country'] = feature_nominal2scalar(data_test['native.country'])
    dataset_test_copy['income'] = feature_nominal2scalar(data_test['income'])
    # =======transform numerical attributes to tensor=========
    age_numeric = numeric_tensor(dataset_test_copy['age'])
    # edu_num_numeric = numeric_tensor(dataset_test_copy['education.num'])
    capgain_numeric = numeric_tensor(dataset_test_copy['capital.gain'])
    caploss_numeric = numeric_tensor(dataset_test_copy['capital.loss'])
    hours_numeric = numeric_tensor(dataset_test_copy['hours.per.week'])
    # =======transform scalar attributes to one hot code======
    edu_one_hot = one_hot_tensor(dataset_test_copy['education'])
    workclass_one_hot = one_hot_tensor(dataset_test_copy['workclass'])
    marital_one_hot = one_hot_tensor(dataset_test_copy['marital.status'])
    occupation_one_hot = one_hot_tensor(dataset_test_copy['occupation'])
    relationship_one_hot = one_hot_tensor(dataset_test_copy['relationship'])
    race_one_hot = one_hot_tensor(dataset_test_copy['race'])
    sex_one_hot = one_hot_tensor(dataset_test_copy['sex'])
    native_one_hot = one_hot_tensor(dataset_test_copy['native.country'])

    income_label = numeric_tensor(dataset_test_copy['income'])

    # =======combine all attributes to dataset=======
    dataset_adult = torch.cat((age_numeric, workclass_one_hot, edu_one_hot, marital_one_hot,
                               occupation_one_hot, relationship_one_hot, race_one_hot, sex_one_hot, capgain_numeric,
                               caploss_numeric, hours_numeric, native_one_hot), 1)
    print(dataset_adult.size())

    if mode == '':
        Mydataset = Data.TensorDataset(dataset_adult, income_label)
    elif mode == 'c':
        # construct a combine-label
        combine_label = torch.cat((income_label, sex_one_hot), 1)
        Mydataset = Data.TensorDataset(dataset_adult, combine_label)

    return Mydataset


def Dataset_maker_resampling(URL, mode=''):
    """
    read data and make the dataset
    """
    # URL = "./data/adult_census_income/adult_normalized.csv"
    dataset_adult = pd.read_csv(URL)
    data_test = dataset_adult[['age', 'workclass', 'education', 'education.num', 'marital.status', 'occupation',
                               'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week',
                               'native.country', 'income']]
    """
    data preprocessing
    """
    # ====== outlier detection and remove ========
    outlier_list = (data_test.loc[(data_test['sex'] == 'Female') & (
            data_test['relationship'] == 'Husband')].index.values).tolist()
    outlier_list.extend((data_test.loc[(data_test['sex'] == 'Male') & (
                data_test['relationship'] == 'Wife')].index.values).tolist())
    for i, index in enumerate(outlier_list):
        data_test.drop([index], inplace=True)
    data_test = data_test.reset_index(drop=True)
    # ======= map income to scalar value ========
    data_test['income'] = data_test['income'].map({'<=50K': 0, '>50K': 1})
    # ======= calculate weight =======
    weight_df = reweighing_calculate(data_test, 'sex', 'income')

    # =======transform nominal attributes to scalar==========
    dataset_test_copy = data_test.copy(deep=True)
    dataset_test_copy['education'] = feature_nominal2scalar(data_test['education'])
    dataset_test_copy['workclass'] = feature_nominal2scalar(data_test['workclass'])
    dataset_test_copy['marital.status'] = feature_nominal2scalar(data_test['marital.status'])
    dataset_test_copy['occupation'] = feature_nominal2scalar(data_test['occupation'])
    dataset_test_copy['relationship'] = feature_nominal2scalar(data_test['relationship'])
    dataset_test_copy['race'] = feature_nominal2scalar(data_test['race'])
    dataset_test_copy['sex'] = feature_nominal2scalar(data_test['sex'])
    dataset_test_copy['native.country'] = feature_nominal2scalar(data_test['native.country'])
    dataset_test_copy['income'] = feature_nominal2scalar(data_test['income'])
    # =======transform numerical attributes to tensor=========
    age_numeric = numeric_tensor(dataset_test_copy['age'])
    edu_num_numeric = numeric_tensor(dataset_test_copy['education.num'])
    capgain_numeric = numeric_tensor(dataset_test_copy['capital.gain'])
    caploss_numeric = numeric_tensor(dataset_test_copy['capital.loss'])
    hours_numeric = numeric_tensor(dataset_test_copy['hours.per.week'])
    # =======transform scalar attributes to one hot code======
    edu_one_hot = one_hot_tensor(dataset_test_copy['education'])
    workclass_one_hot = one_hot_tensor(dataset_test_copy['workclass'])
    marital_one_hot = one_hot_tensor(dataset_test_copy['marital.status'])
    occupation_one_hot = one_hot_tensor(dataset_test_copy['occupation'])
    relationship_one_hot = one_hot_tensor(dataset_test_copy['relationship'])
    race_one_hot = one_hot_tensor(dataset_test_copy['race'])
    sex_one_hot = one_hot_tensor(dataset_test_copy['sex'])
    native_one_hot = one_hot_tensor(dataset_test_copy['native.country'])

    income_label = numeric_tensor(dataset_test_copy['income'])

    # =======combine all attributes to dataset=======
    dataset_adult = torch.cat((age_numeric, workclass_one_hot, edu_one_hot, edu_num_numeric, marital_one_hot,
                               occupation_one_hot, relationship_one_hot, race_one_hot, sex_one_hot, capgain_numeric,
                               caploss_numeric, hours_numeric, native_one_hot, income_label), 1)

    # ======== upsampling ========
    dataset_adult_au, sex_au, income_au = Upsampling(dataset_adult, weight_df)

    if mode == '':
        Mydataset = Data.TensorDataset(dataset_adult_au, income_au)
    elif mode == 'c':
        # construct a combine-label
        combine_label = torch.cat((income_au, sex_au), 1)
        Mydataset = Data.TensorDataset(dataset_adult_au, combine_label)

    return Mydataset


# ================== classification dataset maker ====================
def Dataset_maker_classification(URL):
    encoded_adult_dataset_test = pd.read_csv(URL)
    print(encoded_adult_dataset_test.shape)
    # encoded_adult_dataset_test = encoded_adult_dataset.drop(['race'], axis=1)
    encoded_adult_dataset_test['income'] = encoded_adult_dataset_test['income'].map({'<=50K': 0, '>50K': 1})
    encoded_adult_dataset_test['sex'] = feature_nominal2scalar(encoded_adult_dataset_test['sex'])
    sex_one_hot = one_hot_tensor(encoded_adult_dataset_test['sex'])
    # ======transform numerical attribute to tensor ======
    f1_numeric = numeric_tensor(encoded_adult_dataset_test['feature_1'])
    f2_numeric = numeric_tensor(encoded_adult_dataset_test['feature_2'])
    f3_numeric = numeric_tensor(encoded_adult_dataset_test['feature_3'])
    f4_numeric = numeric_tensor(encoded_adult_dataset_test['feature_4'])
    f5_numeric = numeric_tensor(encoded_adult_dataset_test['feature_5'])
    f6_numeric = numeric_tensor(encoded_adult_dataset_test['feature_6'])
    f7_numeric = numeric_tensor(encoded_adult_dataset_test['feature_7'])
    f8_numeric = numeric_tensor(encoded_adult_dataset_test['feature_8'])
    f9_numeric = numeric_tensor(encoded_adult_dataset_test['feature_9'])
    f10_numeric = numeric_tensor(encoded_adult_dataset_test['feature_10'])
    income_label = numeric_tensor(encoded_adult_dataset_test['income'])
    # =======combine all attributes to dataset =======
    dataset_encoded = torch.cat((f1_numeric, f2_numeric, f3_numeric, f4_numeric, f5_numeric, f6_numeric, f7_numeric,
                                 f8_numeric, f9_numeric, f10_numeric), 1)

    # construct a combine-label
    combine_label = torch.cat((income_label, sex_one_hot), 1)
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
    female_set_num, male_set_num = [], []

    data = pd.read_csv(URL)
    data_test = data.drop(['race'], axis=1)
    # seperate dataset in two gender
    data_female = data_test.loc[data_test['sex'] == 'Female']
    data_male = data_test.loc[data_test['sex'] == 'Male']
    data_female.reset_index(drop=True, inplace=True)
    data_male.reset_index(drop=True, inplace=True)
    # regenerate train, val, test with equal gender distribution
    quantity.append(len(data_female))
    quantity.append(len(data_male))
    for i in range(2):
        train_num = quantity[i] - math.ceil(quantity[i] * proportion[1]) - math.ceil(quantity[i] * proportion[2])
        val_num = math.ceil(quantity[i] * proportion[1])
        test_num = math.ceil(quantity[i] * proportion[2])
        if i == 0:
            female_set_num.append(train_num)
            female_set_num.append(val_num)
            female_set_num.append(test_num)
        elif i == 1:
            male_set_num.append(train_num)
            male_set_num.append(val_num)
            male_set_num.append(test_num)

    # step 2: shuffle female and male dataset respectively
    data_female = data_female.sample(frac=1.0)
    data_female.reset_index(drop=True, inplace=True)
    train_female = data_female.loc[0: female_set_num[0] - 1]
    val_female = data_female.loc[female_set_num[0]: sum(female_set_num[0:2]) - 1]
    test_female = data_female.loc[sum(female_set_num[0:2]): sum(female_set_num)]

    data_male = data_male.sample(frac=1.0)
    data_male.reset_index(drop=True, inplace=True)
    train_male = data_male.loc[0: male_set_num[0] - 1]
    val_male = data_male.loc[male_set_num[0]: sum(male_set_num[0:2]) - 1]
    test_male = data_male.loc[sum(male_set_num[0:2]): sum(male_set_num)]

    # step 3: connect female and male data set together
    train_data = pd.concat([train_female, train_male])
    val_data = pd.concat([val_female, val_male])
    test_data = pd.concat([test_female, test_male])

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
        dataset_list[j]['income'] = dataset_list[j]['income'].map({'<=50K': 0, '>50K': 1})
        dataset_list[j]['sex'] = feature_nominal2scalar(dataset_list[j]['sex'])
        sex_one_hot = one_hot_tensor(dataset_list[j]['sex'])

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
        income_label = numeric_tensor(dataset_list[j]['income'])

        dataset_encoded = torch.cat((f1_numeric, f2_numeric, f3_numeric, f4_numeric, f5_numeric, f6_numeric, f7_numeric,
                                     f8_numeric, f9_numeric, f10_numeric), 1)
        # construct a combine-label
        combine_label = torch.cat((income_label, sex_one_hot), 1)
        Mydataset = Data.TensorDataset(dataset_encoded, combine_label)

        dataset_tensor_list.append(Mydataset)

    return dataset_tensor_list[0], dataset_tensor_list[1], test_data

