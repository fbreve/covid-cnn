# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:00:58 2019

@author: fbrev

based on visually-impaired-aid-tl-finetune

Required packages:
    - numpy
    - pandas
    - tensorflow / tensorflow-gpu
    - pillow
    - scikit-learn
"""

import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#import os
#print(os.listdir("dataset/"))

#FAST_RUN = False
#IMAGE_WIDTH=128
#IMAGE_HEIGHT=128
#IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

ALT_DENSE = True # True, False
POOLING = 'avg' # None, 'avg', 'max'
DATA_AUG = False # True, False
DATA_AUG_MULT = 1 # >=1
OPTIMIZER = 'adam' # 'rmsprop', 'adam', 'sgd', etc.
MODEL_TYPE = 'EfficientNetB2'
N_REPEATS = 10
BATCH_SIZE = 16

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
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        image_size = (224, 224)
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='Xception':
        image_size = (299, 299)
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet50':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet101, preprocess_input
        model = ResNet101(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
        model = ResNet152(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))    
    elif model_type=='ResNet50V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        model = ResNet50V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='ResNet101V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2, preprocess_input
        model = ResNet101V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='ResNet152V2':
        image_size = (224, 224)
        from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
        model = ResNet152V2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionV3':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
        model = InceptionV3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='InceptionResNetV2':
        image_size = (299, 299)
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
        model = InceptionResNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))  
    elif model_type=='MobileNet':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))       
    elif model_type=='DenseNet121':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
        model = DenseNet121(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='DenseNet169':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
        model = DenseNet169(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,)) 
    elif model_type=='DenseNet201':
        image_size = (224, 224)
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetLarge':
        image_size = (331, 331)
        from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
        model = NASNetLarge(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))         
    elif model_type=='NASNetMobile':
        image_size = (224, 224)
        from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
        model = NASNetMobile(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))   
    elif model_type=='MobileNetV2':
        image_size = (224, 224)
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input        
        model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))                 
    elif model_type=='EfficientNetB0':
        image_size = (224, 224)
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB1':
        image_size = (240, 240)
        from tensorflow.keras.applications.efficientnet import EfficientNetB1, preprocess_input
        model = EfficientNetB1(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB2':
        image_size = (260, 260)
        from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
        model = EfficientNetB2(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB3':
        image_size = (300, 300)
        from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
        model = EfficientNetB3(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB4':
        image_size = (380, 380)
        from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
        model = EfficientNetB4(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB5':
        image_size = (456, 456)
        from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
        model = EfficientNetB5(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB6':
        image_size = (528, 528)
        from tensorflow.keras.applications.efficientnet import EfficientNetB6, preprocess_input
        model = EfficientNetB6(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    elif model_type=='EfficientNetB7':
        image_size = (600, 600)
        from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
        model = EfficientNetB7(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))        
    else: print("Error: Model not implemented.")

    preprocessing_function = preprocess_input

    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.models import Model

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
                  
	# add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
 
    if ALT_DENSE==False:
        class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    else:
        dense1 = Dense(256, activation='relu', kernel_initializer='he_uniform')(flat1)
        class1 = Dropout(0.25)(dense1)

    output = Dense(2, activation='softmax')(class1)
   
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
	
    model.summary()
    
    return model, preprocessing_function, image_size

def train_test_model(train_df, test_df, model, preprocessing_function, image_size, rep):
   
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    earlystop = EarlyStopping(patience=10)
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]
    
    train_df["category"] = train_df["category"].replace({1: 'positive', 0: 'negative'}) 
    test_df["category"] = test_df["category"].replace({1: 'positive', 0: 'negative'})        

    # let's split the train set in a small training subset and a large test subset    
    train_df, traintest_df = train_test_split(
        train_df, test_size=0.95, stratify=train_df.category, shuffle=True, random_state=rep)

    # save the train subset for PCC usage (labeled subset)
    with open("trainsubset.csv","a+") as f_csv:
        for item in train_df.index:
            f_csv.write("%i, " % item)
        f_csv.write("\n")

    # now let's create a validation subset
    train_df, validate_df = train_test_split(
        train_df, test_size=0.20, stratify=train_df.category, shuffle=True, random_state=rep)
    
    train_df = train_df.reset_index(drop=True)
    traintest_df = traintest_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    #train_df['category'].value_counts().plot.bar()
    
    #validate_df['category'].value_counts().plot.bar()
    
    total_train = train_df.shape[0]
    total_traintest = traintest_df.shape[0]
    total_validate = validate_df.shape[0]
    total_test = test_df.shape[0]

    if DATA_AUG==True:
        train_datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=(0.9, 1.1),
            zoom_range=(0.85, 1.15),
            fill_mode='constant',
            cval=0.,
            preprocessing_function=preprocessing_function
        )
    else:
        train_datagen = ImageDataGenerator(
            #rescale=1./255,
            preprocessing_function=preprocessing_function
        )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        DATASET_TRAIN_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
    )
    
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        DATASET_TRAIN_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=BATCH_SIZE,        
    )

    #import psutil
    
    #epochs=3 if FAST_RUN else 50
    
    class_weight = {0: 0.5782, 1: 3.6960}
    
    print("Training 5 epochs with convolutional layers weights locked...")
    epochs=5
    model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate/BATCH_SIZE,
        #steps_per_epoch=total_train,
        steps_per_epoch= total_train * DATA_AUG_MULT / BATCH_SIZE,
        callbacks=callbacks,
        #use_multiprocessing=True,
        #workers=psutil.cpu_count()
        class_weight = class_weight

    )
    
    # now let's unlock all the weights
    for layer in model.layers:
        layer.trainable = True

    print("Training up to 45 epochs with all weights unlocked...")
    epochs=45
    model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate/BATCH_SIZE,
        #steps_per_epoch=total_train,
        steps_per_epoch= total_train * DATA_AUG_MULT / BATCH_SIZE,
        callbacks=callbacks,
        #use_multiprocessing=True,
        #workers=psutil.cpu_count()
        class_weight = class_weight
    )    


    # testing with the test_split which is actually 80% of the training data
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    test_generator = test_datagen.flow_from_dataframe(
         traintest_df, 
         DATASET_TRAIN_PATH, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=image_size,
         batch_size=BATCH_SIZE,
         shuffle=False
    )
        
    predictions = model.predict(
        test_generator,
        steps=np.ceil(total_traintest/BATCH_SIZE),
        verbose=1
    )
    y_true = traintest_df["category"].replace({'positive':1, 'negative':0})    
    y_pred = np.argmax(predictions,1)
    cm = confusion_matrix(y_true, y_pred)
    acc = np.trace(cm) / np.sum(cm)
    tp = cm[1,1] # true positive
    fn = cm[1,0] # false negative
    tpr = tp / (tp+fn) # sensitivity, recall, hit rate, or true positive rate (TPR)
    fp = cm[0,1] # false positive
    ppv = tp / (tp + fp); # precision or positive predictive value (PPV)
    f1 = 2*tp / (2*tp + fp + fn) # F1 score   

    # testing with the real test split
    test_generator = test_datagen.flow_from_dataframe(
         test_df, 
         DATASET_TEST_PATH, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=image_size,
         batch_size=BATCH_SIZE,
         shuffle=False
    )    

    predictions = model.predict(
        test_generator, 
        steps=np.ceil(total_test/BATCH_SIZE),
        verbose=1
    )
    y_true = test_df["category"].replace({'positive':1, 'negative':0})    
    y_pred = np.argmax(predictions,1)
    cm = confusion_matrix(y_true, y_pred)
    acc_t = np.trace(cm) / np.sum(cm)
    tp = cm[1,1] # true positive
    fn = cm[1,0] # false negative
    tpr_t = tp / (tp+fn) # sensitivity, recall, hit rate, or true positive rate (TPR)
    fp = cm[0,1] # false positive
    ppv_t = tp / (tp + fp); # precision or positive predictive value (PPV)
    f1_t = 2*tp / (2*tp + fp + fn) # F1 score   

    return acc,tpr,ppv,f1,acc_t,tpr_t,ppv_t,f1_t
    
# Main
    
# get hostname for log-files
import socket
hostname = socket.gethostname()

# create filenames
log_filename = "covid-tl-ssl-" + hostname + ".log"
csv_filename = "covid-tl-ssl-" + hostname + ".csv"

# write log header
with open(log_filename,"a+") as f_log:
    f_log.write("Machine: %s\n" % hostname)
    from datetime import datetime
    now = datetime.now()
    f_log.write(now.strftime("Date: %d/%m/%Y Time: %H:%M:%S\n"))
    f_log.write("Alternative Dense Layer: %s\n" % ALT_DENSE)
    f_log.write("Pooling Application Layer: %s\n" % POOLING)
    f_log.write("Data Augmentation: %s\n" % DATA_AUG)
    f_log.write("Data Augmentation Multiplier: %s\n" % DATA_AUG_MULT)
    f_log.write("Optimizer: %s\n" %OPTIMIZER)
    f_log.write("Model: %s\n" % MODEL_TYPE)
    f_log.write("N Repeats: %s\n" % N_REPEATS)
    f_log.write("Batch Size: %s\n\n" % BATCH_SIZE)
           
# Seed to make it reproducible
np.random.seed(seed=1980)

df_train = load_data(LABELS_TRAIN_FILE)
df_test = load_data(LABELS_TEST_FILE)

model, preprocessing_function, image_size = create_model(MODEL_TYPE)

# save weights before training the model
ini_weights = model.get_weights()        
       
# vector to hold each fold accuracy
scores = []

# enumerate allow the usage of the index for prints
for rep in range(0,N_REPEATS):
    # set the weights to their initial state before each training
    model.set_weights(ini_weights)
    
    # call training, passing repetition number as the random_state to make it reproducibly
    acc, tpr, ppv, f1, acc_t, tpr_t, ppv_t, f1_t = train_test_model(df_train,df_test,model,preprocessing_function,image_size,rep)
    scores.append([acc, tpr, ppv, f1, acc_t, tpr_t, ppv_t, f1_t])

    # print results to screen   
    print("Repetition: %i of %i" % (rep+1, N_REPEATS))
    print("Train Test Subset:  Acc: %.4f TPR: %.4f PPV: %.4f F1: %.4f" % (acc, tpr, ppv, f1))
    print("Actual Test Subset: Acc: %.4f TPR: %.4f PPV: %.4f F1: %.4f\n" % (acc_t, tpr_t, ppv_t, f1_t))
    
    m_acc, m_tpr, m_ppv, m_f1, m_acc_t, m_tpr_t, m_ppv_t, m_f1_t = np.mean(scores,0)
    s_acc, s_tpr, s_ppv, s_f1, s_acc_t, s_tpr_t, s_ppv_t, s_f1_t = np.std(scores,0)
    
    print("Mean Acc: %.4f (+/- %.4f)" % (m_acc, s_acc))
    print("Mean TPR: %.4f (+/- %.4f)" % (m_tpr, s_tpr))
    print("Mean PPV: %.4f (+/- %.4f)" % (m_ppv, s_ppv))
    print("Mean F1:  %.4f (+/- %.4f)\n" % (m_f1, s_f1))    
    
    print("Mean Test Acc: %.4f (+/- %.4f)" % (m_acc_t, s_acc_t))
    print("Mean Test TPR: %.4f (+/- %.4f)" % (m_tpr_t, s_tpr_t))
    print("Mean Test PPV: %.4f (+/- %.4f)" % (m_ppv_t, s_ppv_t))
    print("Mean Test F1:  %.4f (+/- %.4f)\n" % (m_f1_t, s_f1_t))        
    
    
    #record log file
    with open(log_filename,"a+") as f_log:
        f_log.write("Repetition: %i of %i\n" % (rep+1, N_REPEATS))
        f_log.write("Train Test Subset:\n")
        f_log.write("Acc: %.4f Mean: %.4f (+/- %.4f)\n" % (acc, m_acc, s_acc)) 
        f_log.write("TPR: %.4f Mean: %.4f (+/- %.4f)\n" % (tpr,m_tpr,s_tpr))
        f_log.write("PPV: %.4f Mean: %.4f (+/- %.4f)\n" % (ppv,m_ppv,s_ppv))
        f_log.write("F1:  %.4f Mean: %.4f (+/- %.4f)\n" % (f1,m_f1,s_f1))    
        f_log.write("Actual Test Subset:\n")
        f_log.write("Acc: %.4f Mean: %.4f (+/- %.4f)\n" % (acc_t, m_acc_t, s_acc_t)) 
        f_log.write("TPR: %.4f Mean: %.4f (+/- %.4f)\n" % (tpr_t, m_tpr_t ,s_tpr_t))
        f_log.write("PPV: %.4f Mean: %.4f (+/- %.4f)\n" % (ppv_t, m_ppv_t ,s_ppv_t))
        f_log.write("F1:  %.4f Mean: %.4f (+/- %.4f)\n\n" % (f1_t, m_f1_t, s_f1_t))    

    
    #record individual results to csv file
    with open(csv_filename,"a+") as f_csv:
        f_csv.write("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n" % (acc, tpr, ppv, f1, acc_t, tpr_t, ppv_t, f1_t))
        if rep+1==N_REPEATS: f_csv.write("\n")
