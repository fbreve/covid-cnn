# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:25:17 2021

@author: fbrev
"""

import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
import os
import random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1980'
SEED = 1980
random.seed(SEED)
np.random.seed(SEED)

LABELS_TEST_FILE   = "../COVID-Net/labels/test_COVIDx8B.txt"
N_REPEATS = 5
POOLING = 'avg' # None, 'avg', 'max'


def load_data(labels_file):
    
    with open(labels_file, 'r') as f:
        labels = [line.strip().split()[-3:-1] for line in f]
      
    filenames = [row[0] for row in labels]       
    categories = []
    for filename in filenames:
        # gets the label from the label file
        category = labels[[row[0] for row in labels].index(filename)][1]       
        if category == 'positive':
            categories.append(1)
        else:
            categories.append(0)
    
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

# Main

df_test = load_data(LABELS_TEST_FILE)
y_true = df_test["category"].replace({'positive':1, 'negative':0})

#model_type_list = ['DenseNet169', 'EfficientNetB2', ]
#model_type_list = ['DenseNet169', 'EfficientNetB2', 'InceptionResNetV2']
#model_type_list = ['DenseNet169', 'EfficientNetB2', 'InceptionResNetV2', 'InceptionV3']
#model_type_list = ['DenseNet169', 'EfficientNetB2', 'InceptionResNetV2', 'InceptionV3', 'MobileNet']
#model_type_list = ['DenseNet169', 'EfficientNetB2', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'EfficientNetB0']
#model_type_list = ['DenseNet169', 'EfficientNetB2', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'EfficientNetB0', 'EfficientNetB3']
model_type_list =['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2',
                   'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2', 'MobileNet', 'MobileNetV2',
                   'DenseNet121', 'DenseNet169',  'DenseNet201', 'NASNetMobile', 'EfficientNetB0', 'EfficientNetB1',
                   'EfficientNetB2', 'EfficientNetB3']


ensemble_size = len(model_type_list) * N_REPEATS
predictions = np.empty([ensemble_size, 400, 2])
scores = []

for index, model_type in enumerate(model_type_list):
    for rep in range(0,N_REPEATS):    
        pred_filename = "./pred/covid-tl-pred-" + model_type + "-" + POOLING + "-" + str(rep+1) + ".csv"
        predictions[index*N_REPEATS + rep] = np.loadtxt(pred_filename, delimiter=',')
    
mean_pred = np.mean(predictions,0)
y_pred = np.argmax(mean_pred,1)
# calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
acc = np.trace(cm) / np.sum(cm) #accuracy
tp = cm[1,1] # true positive
fn = cm[1,0] # false negative
tpr = tp / (tp+fn) # sensitivity, recall, hit rate, or true positive rate (TPR)
fp = cm[0,1] # false positive
ppv = tp / (tp + fp); # precision or positive predictive value (PPV)
f1 = 2*tp / (2*tp + fp + fn) # F1 score   

# print results to screen   
print("Acc: %.4f TPR: %.4f PPV: %.4f F1: %.4f" % (acc, tpr, ppv, f1))    