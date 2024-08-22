import argparse
import os
from datetime import datetime
from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_curve(fpr, tpr, roc_auc, model_type, prex='all'):
    l = {0:'DHP',  1:'PG', 2:'XH',  3:'ZTH'}
    if not os.path.exists('../result4/roc'):
        os.mkdir('../result4/roc')
    lw = 2
    plt.figure()
    plt.plot(fpr['macro'], tpr['macro'], color='navy', linestyle=':',
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc['macro']))
    colors = cycle(['cyan', 'indigo', 'seagreen', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(4), colors):
        plt.plot(fpr[str(i)], tpr[str(i)], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(l[i], roc_auc[str(i)]))
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(model_type)
    plt.savefig('../result4/roc/' + prex + 'test' + model_type + '_' + 'roc.png', dpi=300)

def polt_confusion_matrix(cm, classes, model_type, name):
    # 绘制混淆矩阵图像
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    ax.set_title(model_type)
    # # 创建colorbar
    # cbar = fig.colorbar(im, ax=ax)
    # # 设置colorbar的刻度标签为空字符串
    # cbar.set_ticks(cbar.get_ticks())  # 获取当前的刻度位置
    # cbar.set_ticklabels(['' for _ in cbar.get_ticks()])  # 将刻度标签设置为空字符串
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, ylabel='True Label', xlabel='Predicted Label')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i-0.15, format(round(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            ax.text(j, i+0.15, f"{cm[i, j] / sum(cm[i]) * 100:.2f}%", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(name, dpi=300)
    plt.show()


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