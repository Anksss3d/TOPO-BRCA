import os
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from analysis import generate_scores
import numpy as np


def generate_train_test(TRAIN_DIR, TEST_DIR, features_range, test_split=None):
    '''
    Function to generate Train and test dataset with desired features.
    :param TRAIN_DIR: Train csv file directory
    :param TEST_DIR: Test csv file directory
    :param features_range: Features indices as list.
    :param test_split: Test split, custom if rerquired. By-default the dataset is split into 70:30 split
    :return:
    '''
    df_train = pd.DataFrame(pd.read_csv(TRAIN_DIR, header=None))
    df_test = pd.DataFrame(pd.read_csv(TEST_DIR, header=None))
    print(f"Original Train Shape: {df_train.shape}")
    print(f"Original Test Shape: {df_test.shape}")
    if test_split:
        df = pd.concat([df_train, df_test], axis=0)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df.iloc[:, features_range], df.iloc[:, -1], test_size=test_split, random_state=42)
    else:
        print("Coming here...")
        x_train, y_train = df_train.iloc[:, features_range], df_train.iloc[:, -1]
        x_test, y_test = df_test.iloc[:, features_range], df_test.iloc[:, -1]
    print(f"New Train Shape: {x_train.shape}")
    print(f"New Test Shape: {x_test.shape}")

    return x_train, y_train, x_test, y_test


def random_forest(x_train, y_train, x_test, y_test):
    '''
    Function to train Random Forest model
    :param x_train: Dataframe
    :param y_train: Dataframe
    :param x_test: Dataframe
    :param y_test: Dataframe
    :return: Accuracy of the model
    '''
    model_1 = RandomForestClassifier(n_estimators=300, criterion='entropy',
                                     min_samples_split=10, random_state=0)
    model_1.fit(x_train, y_train)
    predictions = model_1.predict(x_test)
    acc = accuracy_score(y_test, predictions) * 100
    return round(acc, 2)


def xg_boost_kfold(x_train, y_train, x_test, y_test, kwargs, k=5):
    '''
    Train XGBoost model with K-Fold Cross Validation and display all reports
    :param x_train: Dataframe
    :param y_train: Dataframe
    :param x_test: Dataframe
    :param y_test: Dataframe
    :param kwargs: Parameters to the model as dictionary
    :param k: Value of k in K fold
    :return: average accuracy
    '''
    X = pd.concat([x_train, x_test], axis=0)
    y = pd.concat([y_train, y_test], axis=0)
    skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
    split = skf.split(X, y)
    accs, aucs, reports = [None for _ in range(k)], [None for _ in range(k)], [None for _ in range(k)]
    for i, (train_index, test_index) in enumerate(split):
        x_train, y_train = X.iloc[train_index,:], y.iloc[train_index]
        x_test, y_test = X.iloc[test_index, :], y.iloc[test_index]
        modelx = XGBClassifier(**kwargs)
        modelx.fit(x_train, y_train)
        y_predx = modelx.predict(x_test)
        acc = round(accuracy_score(y_test, y_predx) * 100, 2)
        # auc_score = roc_auc_score(y_test, modelx.predict_proba(x_test), average='weighted', multi_class='ovr')
        # cm = confusion_matrix(y_test, y_predx)
        # report = generate_scores(cm)
        #
        # # print(f"Fold {i+1}, Acc: {acc}, AUC: {auc_score}, \nReport: \n {report}")
        # aucs[i]=auc_score
        accs[i] = acc
        # reports[i] = report
    # print(aucs, accs, )
    # avg_auc = sum(aucs)/k
    avg_acc = sum(accs)/k
    # total_report =reports[0]
    # for i in range(1, k):
    #     total_report = total_report.add(reports[i], fill_value=0)
    # total_report = total_report.div(5)
    # print(f"Average-----------Acc: {avg_acc}, avg_AUC: {avg_auc}, \navg_Report: \n {total_report}")
    return round(avg_acc, 2)


def xgboost_kfold(TRAIN_DIR, TEST_DIR, features_range, classes=None):
    '''
    Function to call XGboost K Fold.
    :param TRAIN_DIR: Train csv file directory
    :param TEST_DIR: Test csv file directory
    :param features_range: Features indices as list.
    :return: Average Accuracy of the models
    '''
    x_train, y_train, x_test, y_test = generate_train_test(TRAIN_DIR, TEST_DIR, features_range, classes=classes)
    return xg_boost_kfold(
        x_train,
        y_train,
        x_test,
        y_test,
        kwargs = {'base_score': 0.5, 'colsample_bylevel': 0.5, 'colsample_bytree': 0.5, 'eval_metric': 'mlogloss', 'learning_rate': 0.3, 'max_depth': 9, 'n_estimators': 100, 'subsample': 0.75},
        k=5
    )



def xg_boost_ft(x_train, y_train, x_test, y_test, kwargs):
    '''
    Function to train Simple XGBoost Model
    :param x_train: Dataframe
    :param y_train: Dataframe
    :param x_test: Dataframe
    :param y_test: Dataframe
    :param kwargs: Model Parameters as Dictionary
    :return:
    '''
    print("Keywords received: ", kwargs)
    modelx = XGBClassifier(**kwargs)

    modelx.fit(x_train, y_train)
    y_predx = modelx.predict(x_test)
    acc = round(accuracy_score(y_test, y_predx) * 100, 2)
    # auc_score = roc_auc_score(y_test, modelx.predict_proba(x_test), average='macro', multi_class='ovr')
    # cm = confusion_matrix(y_test, y_predx)
    # report = generate_scores(cm)
    #
    # print(f"Accuracy: {acc}, auc_score : {auc_score}, \nReport: \n{report}")

    return acc, acc, 500


def get_accuracies(FILE_NAME=None):
    '''
    Load Accuracy for different fine-tuning configurations from a CSV file
    :param FILE_NAME: File name if exists
    :return: Accuracy values as list
    '''
    isExist = os.path.exists(FILE_NAME)
    if isExist:
        df = pd.DataFrame(pd.read_csv(FILE_NAME))
        accuracies = df.iloc[1:, 1:].values.tolist()
        print(accuracies)
    else:
        accuracies = []
    return accuracies


def finetune_xgboost(TRAIN_DIR, TEST_DIR, features_range, FILE_NAME="finetuning.csv", DS_NAME="NoName", classes=None):
    '''
    Function to fine-tune XGBoost model with different Parameters
    :param TRAIN_DIR: Train csv file directory
    :param TEST_DIR: Test csv file directory
    :param features_range: Features indices as list.
    :param FILE_NAME: File name with accuracy values
    :param DS_NAME: Dataset Name
    :return: None
    '''
    x_train, y_train, x_test, y_test = generate_train_test(TRAIN_DIR, TEST_DIR, features_range, classes=classes)
    print("x_train shape", x_train.shape)
    accuracies = get_accuracies(FILE_NAME)
    parameters = {
        "max_depth": list(range(5, 10)),
        "learning_rate": [0.277, 0.3, 0.333, 0.4, 0.5],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
        "colsample_bylevel": [1.0],
        "n_estimators": [100, 200, 300, 500, 1000]
    }
    keys = sorted(parameters.keys())
    params = {
        "base_score": 0.5,
        "eval_metric": 'mlogloss',
    }
    tc = 1
    for key in keys:
        tc *= len(parameters[key])
    print(f"Total Combinations : {tc}")
    best_acc = 0
    i=1
    for val10 in parameters[keys[0]]:
        params[keys[0]]=val10
        for val1 in parameters[keys[1]]:
            params[keys[1]] = val1
            for val2 in parameters[keys[2]]:
                params[keys[2]]=val2
                for val3 in parameters[keys[3]]:
                    params[keys[3]] = val3
                    for val4 in parameters[keys[4]]:
                        params[keys[4]] = val4
                        for val5 in parameters[keys[5]]:
                            params[keys[5]] = val5
                            acc = xg_boost_kfold(x_train, y_train, x_test, y_test, kwargs=params, k=5)
                            best_acc = max(acc, best_acc)
                            print(f"{i}/{tc}\tkw={params}\t acc={acc}\t best: {best_acc}")
                            i+=1
                            accuracies.append([str(params), acc])

                            pd.DataFrame(accuracies).to_csv(FILE_NAME)

    print("Finetuning Successfully completed")
    return best_acc



def generate_best_k_features_dataset(TRAIN_DIR, TEST_DIR, features_range, OUTPUT_TRAIN_DIR, OUTPUT_TEST_DIR, k=500):
    '''
    Function to generate the dataset with best k features.
    :param TRAIN_DIR: Train csv file directory
    :param TEST_DIR: Test csv file directory
    :param features_range: Features indices as list.
    :param OUTPUT_TRAIN_DIR: Output Train csv file directory
    :param OUTPUT_TEST_DIR: Output Test csv file directory
    :return:
    '''
    x_train, y_train, x_test, y_test = generate_train_test(TRAIN_DIR, TEST_DIR, features_range, test_split=0.2)
    kwargs = {"base_score": 0.5,"eval_metric": 'mlogloss',}
    model = XGBClassifier(**kwargs)
    model.fit(x_train, y_train)
    importance = model.feature_importances_
    preds = model.predict(x_test)
    print(f"Basic Accuracy: {accuracy_score(y_test, preds)}")
    k = 500
    rimp = sorted(importance, reverse=True)
    thresh = rimp[k]
    support = [i for i in range(len(importance)) if importance[i] > thresh]
    train_df = pd.concat([x_train.iloc[:, support], y_train], axis=1)
    train_df.columns = [i for i in range(train_df.shape[1])]
    test_df = pd.concat([x_test.iloc[:, support], y_test], axis=1)
    test_df.columns = [i for i in range(test_df.shape[1])]
    train_df.to_csv(OUTPUT_TRAIN_DIR)
    test_df.to_csv(OUTPUT_TEST_DIR)

features_ranges = [
    list(range(1, 101)),
    list(range(101, 201)),
    # list(range(1, 201)),
    # list(range(1, 351)),
]
dirs = [
    "MASS",
    "CALC",
]

dirs2 = [
    "CC",
    "MLO"
]

# classes = [
#     None,
#     [0, 1],
#     [1, 2],
#     [0, 2]
# ]
# answers=[]
# answers2 = []
# for features_range in features_ranges:
#     a = []
#     for clases in classes:
#         acc = finetune_xgboost(
#             TRAIN_DIR=r"D://Nisha/dataset1_all_data_350_features_224x224/train/data_2800.csv",
#             TEST_DIR=r"D://Nisha/dataset1_all_data_350_features_224x224/validation/data_2800.csv",
#             features_range=features_range,
#             FILE_NAME=r"D://Nisha/dataset1_all_data_350_features_224x224/finetune.csv",
#             classes=clases
#         )
#         answers.append(f"Feature Range: {min(features_range)}-{max(features_range)}\tClasses: {clases} \tAccuracy: {acc}")
#         a.append(acc)
#     answers2.append(a)
# for answer in answers:
#     print(answer)
# n = np.array(answers2)
# print(n)
# for p in answers2:
#     for i in p:
#         print(i, end=" ")
#     print("")


finetune_xgboost(
        TRAIN_DIR=r"/Users/anksss3d/datasets/breast-cesm/MASS/MLO_200_features/train/data_2800.csv",
        TEST_DIR=r"/Users/anksss3d/datasets/breast-cesm/MASS/MLO_200_features/validation/data_2800.csv",
        features_range=list(range(1, 201)),
        FILE_NAME=r"/Users/anksss3d/datasets/breast-cesm/MASS/MLO_200_features/finetune.csv",
        classes={
            "benign": 0,
            "malignant": 1,
        },
)
