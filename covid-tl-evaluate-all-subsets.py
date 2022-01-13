# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:59:42 2022

@author: fbrev

based on covid-tl

Required packages:
    - numpy
    - pandas
    - tensorflow / tensorflow-gpu
    - pillow
    - scikit-learn
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import random
#import psutil
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

#FAST_RUN = False
#IMAGE_WIDTH=128
#IMAGE_HEIGHT=128
#IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

ALT_DENSE = True # True, False
POOLING = 'avg' # None, 'avg', 'max'
DATA_AUG = False # True, False
MODEL_TYPE = 'ResNet101V2'
N_REPEATS = 5
BATCH_SIZE = 16
MULTI_OPTIMIZER = True

DATASET_TRAIN_PATH = "../COVID-Net/data/train/"
DATASET_TEST_PATH  = "../COVID-Net/data/test/"
LABELS_TRAIN_FILE  = "../COVID-Net/labels/train_COVIDx8B.txt"
LABELS_TEST_FILE   = "../COVID-Net/labels/test_COVIDx8B.txt"

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

def create_model(model_type):
    # load model and preprocessing_function    
    if model_type=='VGG16':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='Xception':
        image_size = (299, 299)
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet50':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        model = ResNet50(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        model = ResNet101(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        model = ResNet152(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        model = ResNet50V2(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
        model = ResNet101V2(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        model = ResNet152V2(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        model = InceptionResNetV2(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        model = DenseNet121(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        model = DenseNet169(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
        model = NASNetMobile(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input        
        model = MobileNetV2(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                 
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
        model = EfficientNetB1(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
        model = EfficientNetB2(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        model = EfficientNetB3(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
        model = EfficientNetB4(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
        model = EfficientNetB5(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
        model = EfficientNetB6(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        model = EfficientNetB7(weights=None, include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    else: print("Error: Model not implemented.")

    preprocessing_function = preprocess_input
    
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.models import Model
                 
	# add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
 
    initializer = tf.keras.initializers.HeUniform(seed=SEED)
    if ALT_DENSE==False:
        class1 = Dense(128, activation='relu', kernel_initializer=initializer)(flat1)
    else:
        dense1 = Dense(256, activation='relu', kernel_initializer=initializer)(flat1)
        class1 = Dropout(0.2)(dense1)

    output = Dense(2, activation='softmax')(class1)
   
    # define new model
    model = Model(inputs=model.inputs, outputs=output)


    # compile model
    
    if MULTI_OPTIMIZER==True:
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate=1e-5),
            tf.keras.optimizers.Adam(learning_rate=1e-3)
            ]
        optimizers_and_layers = [(optimizers[0], model.layers[0:-4]), (optimizers[1], model.layers[-4:])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    else:    
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
       
    return model, preprocessing_function, image_size

def evaluate_model(df, model, preprocessing_function, image_size, rep, dataset_path):
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    generator = test_datagen.flow_from_dataframe(
         df, 
         directory=dataset_path, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=image_size,
         batch_size=BATCH_SIZE,
         shuffle=False,
         seed=SEED+rep,
    )
           
    #_, acc = model.evaluate(test_generator, steps=np.ceil(total_test/BATCH_SIZE))
    predictions = model.predict(
        generator,
        batch_size=BATCH_SIZE,
        verbose=1
    )    

    # save predictions to file inside the 'pred' subfolder
    pred_filename = "./pred/covid-tl-pred-" + MODEL_TYPE + "-" + str(POOLING) + "-" + str(rep+1) + ".csv"
    np.savetxt(pred_filename, predictions, delimiter=',')

    # ground truth
    y_true = df["category"].replace({'positive':1, 'negative':0})
    # get hard labels from the predictions
    y_pred = np.argmax(predictions,1)
    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / np.sum(cm) #accuracy
    tp = cm[1,1] # true positive
    fn = cm[1,0] # false negative
    tpr = tp / (tp+fn) # sensitivity, recall, hit rate, or true positive rate (TPR)
    fp = cm[0,1] # false positive
    ppv = tp / (tp + fp) # precision or positive predictive value (PPV)
    f1 = 2*tp / (2*tp + fp + fn) # F1 score 
    return acc,tpr,ppv,f1

def test_model(train_df, test_df, model, preprocessing_function, image_size, rep):
   
    train_df["category"] = train_df["category"].replace({1: 'positive', 0: 'negative'}) 
    test_df["category"] = test_df["category"].replace({1: 'positive', 0: 'negative'})        
    
    train_df, validate_df = train_test_split(
        train_df, test_size=0.20, stratify=train_df.category, shuffle=True, random_state=SEED+rep)
    
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    acc = np.empty(3)
    tpr = np.empty(3)
    ppv = np.empty(3)
    f1 = np.empty(3)
    acc[0],tpr[0],ppv[0],f1[0] = evaluate_model(train_df,model, preprocessing_function, image_size, rep, DATASET_TRAIN_PATH)
    acc[1],tpr[1],ppv[1],f1[1] = evaluate_model(validate_df, model, preprocessing_function, image_size, rep, DATASET_TRAIN_PATH)
    acc[2],tpr[2],ppv[2],f1[2] = evaluate_model(test_df,model, preprocessing_function, image_size, rep, DATASET_TEST_PATH)        
      
    return acc,tpr,ppv,f1
    
# Main
from time import perf_counter
t1_start = perf_counter()
    
# get hostname for log-files
import socket
hostname = socket.gethostname()

model_type_list = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',  
                   'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2',
                   'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile',
                   'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3']

# create filenames
acc_file = "covid-tl-acc-" + str(POOLING) + ".csv"
tpr_file = "covid-tl-tpr-" + str(POOLING) + ".csv"
ppv_file = "covid-tl-ppv-" + str(POOLING) + ".csv"
f1_file =  "covid-tl-f1-"  + str(POOLING) + ".csv"
         
df_train = load_data(LABELS_TRAIN_FILE)
df_test = load_data(LABELS_TEST_FILE)

tab_acc = np.zeros([21,3])
tab_tpr = np.zeros([21,3])
tab_ppv = np.zeros([21,3])
tab_f1 = np.zeros([21,3])

for model_index, model_type in enumerate(model_type_list):
    model, preprocessing_function, image_size = create_model(model_type)
    acc = np.empty([N_REPEATS,3])
    tpr = np.empty([N_REPEATS,3])
    ppv = np.empty([N_REPEATS,3])
    f1 = np.empty([N_REPEATS,3])
    
    for rep in range(0,N_REPEATS):
        # load model
        model_filename = "./weights/covid-tl-weights-" + model_type + "-" + str(POOLING) + "-" + str(rep+1) + ".h5"
        model.load_weights(model_filename)
        acc[rep], tpr[rep], ppv[rep], f1[rep] = test_model(df_train,df_test,model,preprocessing_function,image_size,rep)
        print("Model: %s Rep.: %i/%i" % (model_type, rep+1, N_REPEATS))    
        print("ACC [Train, Val, Test]: ", acc[rep])
        print("TPR [Train, Val, Test]: ", tpr[rep])
        print("PPV [Train, Val, Test]: ", ppv[rep])
        print("F1  [Train, Val, Test]: ", f1[rep])
    
    tab_acc[model_index] = np.mean(acc,0)
    tab_tpr[model_index] = np.mean(tpr,0)
    tab_ppv[model_index] = np.mean(ppv,0)
    tab_f1[model_index] = np.mean(f1,0)
    
    print("Model: %s" % (model_type))    
    print("ACC [Train, Val, Test]: ", tab_acc[model_index])
    print("TPR [Train, Val, Test]: ", tab_tpr[model_index])
    print("PPV [Train, Val, Test]: ", tab_ppv[model_index])
    print("F1  [Train, Val, Test]: ", tab_f1[model_index])
    
    np.savetxt(acc_file, tab_acc, delimiter=',', fmt='%0.4f')
    np.savetxt(tpr_file, tab_tpr, delimiter=',', fmt='%0.4f')
    np.savetxt(ppv_file, tab_ppv, delimiter=',', fmt='%0.4f')
    np.savetxt(f1_file,  tab_f1,  delimiter=',', fmt='%0.4f')
   
t1_stop = perf_counter()
t1_elapsed = t1_stop-t1_start
print("Elapsed time: ",t1_elapsed)