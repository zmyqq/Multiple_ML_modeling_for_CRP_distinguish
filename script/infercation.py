import argparse
from itertools import cycle

import matplotlib
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, confusion_matrix, accuracy_score, \
    matthews_corrcoef, classification_report, auc, roc_curve
import os
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from script.utils import polt_confusion_matrix, plot_curve
matplotlib.use('agg')  # 设置后端为agg
result_dict = {}
# 训练每个分类器并获取特征重要性
feature_importances = {}


# 计算特异度和阴性预测值
def calculate_specificity_npv(predictions, true_labels, class_label):
    # 将预测结果和真实标签转换为NumPy数组
    y_pred = np.array(predictions)
    y_true = np.array(true_labels)

    # 根据类别标签获取正例和负例的索引
    positive_indices = np.where(y_true == class_label)[0]
    negative_indices = np.where(y_true != class_label)[0]

    # 计算特异度
    true_negatives = np.sum(y_pred[negative_indices] == y_true[negative_indices])
    false_positives = np.sum(y_pred[positive_indices] != y_true[positive_indices])
    specificity = true_negatives / (true_negatives + false_positives)

    # 计算阴性预测值
    false_negatives = np.sum(y_pred[negative_indices] != y_true[negative_indices])
    negative_predictive_value = true_negatives / (true_negatives + false_negatives)

    return specificity, negative_predictive_value



# 定义评价指标的函数
def evaluate_model(model, X, y, number, model_type):
    if not os.path.exists('../result4/confusion_matrix'):
        os.mkdir('../result4/confusion_matrix')
    # 计算每个类别的AUC
    fpr = dict()
    tpr = dict()
    n_classes = 4
    roc_auc = dict()
    y_pred = model.predict(X)
    polt_confusion_matrix(confusion_matrix(y, y_pred), classes=['DHP', 'PG', 'XH', 'ZTH'], model_type=model_type,
                          name='../result4/confusion_matrix/' + model_type + 'confusion_matrix.png')
    # 计算每个类别的Specificity, npv
    specificities, npvs = [], []
    for i in range(len(np.unique(y))):
        sp, npv = calculate_specificity_npv(y_pred, y, i)
        specificities.append(sp)
        npvs.append(npv)
    # 计算recall
    report = classification_report(y, y_pred, output_dict=True)
    all_sensitivity = dict()
    for i in range(4):
        all_sensitivity[str(i)] = report[str(i)]['recall']

    probability = model.predict_proba(X)
    accuracy = accuracy_score(y, y_pred)  # 准确率
    sensitivity = recall_score(y, y_pred, labels=np.arange(number), average='weighted')  # 敏感性
    # recall
    for i in range(n_classes):
        fpr[str(i)], tpr[str(i)], _ = roc_curve(y == i, probability[:, i])
        roc_auc[str(i)] = roc_auc_score(y == i, probability[:, i])
    all_fpr = np.unique(np.concatenate([fpr[str(i)] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[str(i)], tpr[str(i)])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    precision = precision_score(y, y_pred, labels=np.arange(number), average='weighted')  # 精准度
    f1 = f1_score(y, y_pred, average='weighted')  # F1
    mcc = matthews_corrcoef(y, y_pred)  # 皮尔森相关系数
    return accuracy, sensitivity, np.mean(specificities), roc_auc, precision, f1, mcc, np.mean(npvs), y_pred, fpr, tpr, all_sensitivity

def infer(file_name, model_type, classfication_num):
    df = pd.read_csv(file_name, header=0)
    data = df.replace({'DHP': 0, 'PG': 1, 'XH': 2, 'ZTH': 3}).values
    # features = data[::, 2::]
    # 选择特定列数据
    selected_columns = [4, 9, 10, 11]
    features = df.iloc[:, selected_columns].values
    # 标准化
    scaler_standard = StandardScaler()
    features = scaler_standard.fit_transform(features)
    labels = data[::, 1].astype(np.int32)

    model = joblib.load(f'../models/{model_type}.m')

    accuracy, sensitivity, specificity, auc, precision, f1, mcc, npv, test_predict, fpr, tpr, all_sensitivity = evaluate_model(model, features,
                                                                                     labels, number=classfication_num, model_type=model_type)

    result_dict[model_type] = {'accuracy': accuracy, 'specificity': specificity, 'auc': auc['macro'],
                               'DHP_auc':auc['0'], 'PG_auc':auc['1'], 'XH_auc':auc['2'], 'ZTH_auc':auc['3'],
                   'precision': precision, 'f1': f1, 'mcc': mcc, 'npv': npv, 'DHP_sensitivity': all_sensitivity['0'],
                               'PG_sensitivity': all_sensitivity['1'], 'XH_sensitivity': all_sensitivity['2'], 'ZTH_sensitivity': all_sensitivity['3'],
                               'sensitivity': sensitivity,}
    plot_curve(fpr, tpr, auc, model_type, prex='one')
    if hasattr(model, 'feature_importances_'):
        # 对于基于树的算法（如RandomForest），使用feature_importances_属性
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 对于线性模型（如LogisticRegression），使用coef_属性的绝对值作为特征重要性
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        # 对于其他算法，无法直接获取特征重要性，将其设为0
        result = permutation_importance(model, features, labels, n_repeats=10, random_state=1, scoring='accuracy')
        importances = result.importances_mean

    feature_importances[model_type] = importances


if __name__ == '__main__':
    model_types = ['LogisticRegression', 'SVM', 'KNN', 'AdaBoost', 'RandomForest', 'LDA', 'QDA', 'ANN', 'naive_bayes']
    file_name = '../data/data.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', default=model_types, type=list)
    parser.add_argument('--file_name', default=file_name, type=str)
    parser.add_argument('--classfication_num', default=4, type=int)
    args = parser.parse_args()
    for model_type in args.model_types:
        infer(file_name=args.file_name, model_type=model_type, classfication_num=args.classfication_num)
    if not os.path.exists('../result4/best_model'):
        os.mkdir('../result4/best_model')
    # 创建一个Excel写入器
    writer = pd.ExcelWriter('../result4/best_model/feature_importances.xlsx')
    # 将每个分类器的特征重要性写入不同的sheet
    for clf_name, importances in feature_importances.items():
        df = pd.DataFrame(importances, columns=['importance'])
        df.to_excel(writer, sheet_name=clf_name, index=False)
    writer.close()
    # 创建一个Excel写入器
    writer = pd.ExcelWriter('../result4/best_model/performance_measures.xlsx')
    # 将每个分类器的特征重要性写入不同的sheet
    for clf_name, value in result_dict.items():
        df = pd.DataFrame(value, index=['best'])
        df.to_excel(writer, sheet_name=clf_name, index=False)
    writer.close()
