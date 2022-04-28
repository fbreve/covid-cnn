# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:59:42 2022

@author: Fabricio Breve

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

# Dataset Paths:
DATASET_TRAIN_PATH = "../data/COVID-Net/data/train/"
DATASET_TEST_PATH  = "../data/COVID-Net/data/test/"
LABELS_TRAIN_FILE  = "../data/COVID-Net/labels/train_COVIDx8B.txt"
LABELS_TEST_FILE   = "../data/COVID-Net/labels/test_COVIDx8B.txt"

# Set RETRAIN to False to use the same weights used to get the published
# results. Set it to True to train the networks again. Warning: it takes a long
# time to retrain.
RETRAIN = True

# Set USE_PUBLISHED_WEIGHTS to True to use the same weights that were used
# in the published results. Set to False to use weights from a previous run.
# Notice that when RETRAIN is set to True, the weights from the retraining
# phase are always used.
USE_PUBLISHED_WEIGHTS = True

# Some alternatives that were tested in earlier stages
ALT_DENSE = True # True, False. True is used in the published results
POOLING = 'avg' # None, 'avg', 'max'. 'avg' is used in the published results
DATA_AUG = False # True, False. False is used in the published results
N_REPEATS = 5 # 5 is used in the published results
BATCH_SIZE = 16 # 16 is used in the published results
MULTI_OPTIMIZER = True # True is used in the published results

MODEL_TYPE_LIST = ['Xception', 'VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',  
                   'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'InceptionV3', 'InceptionResNetV2',
                   'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile',
                   'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3']

IMAGE_CHANNELS=3

FAST_RUN = False  # Set it to True to run only 3 epochs (for debug only)

# Set seeds. 
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

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
    model = Model(inputs=model.inputs, outputs=output, name=model_type)


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
    pred_filename = "../results/predictions/covid-tl-pred-" + model.name + "-" + str(POOLING) + "-" + str(rep+1) + ".csv"
    os.makedirs(os.path.dirname(pred_filename), exist_ok=True)
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

def prepare_subsets(train_df, test_df, rep):
    train_df["category"] = train_df["category"].replace({1: 'positive', 0: 'negative'}) 
    test_df["category"] = test_df["category"].replace({1: 'positive', 0: 'negative'})        
    
    train_df, validate_df = train_test_split(
        train_df, test_size=0.20, stratify=train_df.category, shuffle=True, random_state=SEED+rep)
    
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, validate_df, test_df

def test_model(train_df, test_df, model, preprocessing_function, image_size, rep):
   
    train_df, validate_df, test_df = prepare_subsets(train_df, test_df,rep)
    
    acc = np.empty(3)
    tpr = np.empty(3)
    ppv = np.empty(3)
    f1 = np.empty(3)
    acc[0],tpr[0],ppv[0],f1[0] = evaluate_model(train_df,model, preprocessing_function, image_size, rep, DATASET_TRAIN_PATH)
    acc[1],tpr[1],ppv[1],f1[1] = evaluate_model(validate_df, model, preprocessing_function, image_size, rep, DATASET_TRAIN_PATH)
    acc[2],tpr[2],ppv[2],f1[2] = evaluate_model(test_df,model, preprocessing_function, image_size, rep, DATASET_TEST_PATH)        
      
    return acc,tpr,ppv,f1

def print_and_log(arg):
    print(arg)
    with open(log_filename,"a+") as f_log:
        f_log.write(arg + "\n")

def train_test_model(train_df, test_df, model, preprocessing_function, image_size, rep):
   
    from tensorflow.keras.callbacks import EarlyStopping

    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    callbacks = [earlystop]

    train_df, validate_df, test_df = prepare_subsets(train_df, test_df, rep)        
   
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
            preprocessing_function=preprocessing_function
        )
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        directory=DATASET_TRAIN_PATH, 
        x_col='filename',
        y_col='category',
        target_size=image_size,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        seed=SEED+rep,
    )
    
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    
    validation_generator = validation_datagen.flow_from_dataframe(
         validate_df, 
         directory=DATASET_TRAIN_PATH, 
         x_col='filename',
         y_col='category',
         class_mode='categorical',
         target_size=image_size,
         batch_size=BATCH_SIZE,
         seed=SEED+rep,
    )
  
    class_weight = {0: 0.5782, 1: 3.6960}
    epochs=3 if FAST_RUN else 50
    
    history = model.fit(
        x=train_generator, 
        validation_data=validation_generator,
        epochs=epochs,        
        callbacks=callbacks,
        #use_multiprocessing=True,
        #workers=psutil.cpu_count(),
        class_weight = class_weight,
    )      

    acc = np.empty(3)
    tpr = np.empty(3)
    ppv = np.empty(3)
    f1 = np.empty(3)
    acc[0],tpr[0],ppv[0],f1[0] = evaluate_model(train_df,model, preprocessing_function, image_size, rep, DATASET_TRAIN_PATH)
    acc[1],tpr[1],ppv[1],f1[1] = evaluate_model(validate_df, model, preprocessing_function, image_size, rep, DATASET_TRAIN_PATH)
    acc[2],tpr[2],ppv[2],f1[2] = evaluate_model(test_df,model, preprocessing_function, image_size, rep, DATASET_TEST_PATH)        
          
    # save model
    model_filename = "../results/weights/covid-tl-weights-" + model.name + "-" + str(POOLING) + "-" + str(rep+1) + ".h5"
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    model.save_weights(model_filename)
   
    # save history
    history_filename = "../results/history/covid-tl-history-" + model.name + "-" + str(POOLING) + "-" + str(rep+1) + ".npy"
    os.makedirs(os.path.dirname(history_filename), exist_ok=True)
    np.save(history_filename,history.history)
    
    return acc,tpr,ppv,f1

def train_test_all_models():
      
    if (RETRAIN==True):
        print_and_log("Training and Testing all models:\n")
    
        print_and_log("Alternative Dense Layer: %s" % ALT_DENSE)
        print_and_log("Pooling Application Layer: %s" % POOLING)
        print_and_log("Data Augmentation: %s" % DATA_AUG)
        print_and_log("N Repeats: %s" % N_REPEATS)
        print_and_log("Batch Size: %s" % BATCH_SIZE)
        print_and_log("Multi Optimizer: %s\n" % MULTI_OPTIMIZER)
    else:
        print_and_log("Testing all models in the Training, Validation, and Test subsets:\n")
               
    # Create CSV filenames to hold ACC, TPR, PPV, and F1 results
    acc_file = "../results/covid-tl-acc-" + str(POOLING) + ".csv"
    tpr_file = "../results/covid-tl-tpr-" + str(POOLING) + ".csv"
    ppv_file = "../results/covid-tl-ppv-" + str(POOLING) + ".csv"
    f1_file =  "../results/covid-tl-f1-"  + str(POOLING) + ".csv"
    
    # Load label files
    df_train = load_data(LABELS_TRAIN_FILE)
    df_test = load_data(LABELS_TEST_FILE)
    
    # Matrixes to save ACC, TPR, PPV and F1.
    # Each line corresponds to a model.
    # Columns correspond to Train, Validation, and Test subsets
    mtl_size = len(MODEL_TYPE_LIST)
    tab_acc = np.zeros([mtl_size,3])
    tab_tpr = np.zeros([mtl_size,3])
    tab_ppv = np.zeros([mtl_size,3])
    tab_f1 = np.zeros([mtl_size,3])    
    
    # Repeat for each model
    for model_index, model_type in enumerate(MODEL_TYPE_LIST):
        model, preprocessing_function, image_size = create_model(model_type)
        acc = np.empty([N_REPEATS,3])
        tpr = np.empty([N_REPEATS,3])
        ppv = np.empty([N_REPEATS,3])
        f1 = np.empty([N_REPEATS,3])    

        # save weights before training the model
        ini_weights = model.get_weights()        
                  
        # Repeat for each set of model weights
        for rep in range(0,N_REPEATS):
            
            if (RETRAIN==True):
                # set the weights to their initial state before each training
                model.set_weights(ini_weights)            
                # Test model
                acc[rep], tpr[rep], ppv[rep], f1[rep] = train_test_model(df_train,df_test,model,preprocessing_function,image_size,rep)
            else:
                # Load model
                if (USE_PUBLISHED_WEIGHTS==True):
                    model_filename = "../data/weights/covid-tl-weights-" + model_type + "-" + str(POOLING) + "-" + str(rep+1) + ".h5"
                else:
                    model_filename = "../results/weights/covid-tl-weights-" + model_type + "-" + str(POOLING) + "-" + str(rep+1) + ".h5"
                # Load model weights
                model.load_weights(model_filename)
                # Test model
                acc[rep], tpr[rep], ppv[rep], f1[rep] = test_model(df_train,df_test,model,preprocessing_function,image_size,rep)


            print("Model: %s Rep.: %i/%i" % (model_type, rep+1, N_REPEATS))    
            print("ACC [Train, Val, Test]: ", acc[rep])
            print("TPR [Train, Val, Test]: ", tpr[rep])
            print("PPV [Train, Val, Test]: ", ppv[rep])
            print("F1  [Train, Val, Test]: ", f1[rep])
        
        # Save the ACC, TPR, PPV, and F1 averages for each model
        tab_acc[model_index] = np.mean(acc,0)
        tab_tpr[model_index] = np.mean(tpr,0)
        tab_ppv[model_index] = np.mean(ppv,0)
        tab_f1[model_index] = np.mean(f1,0)
        
        # Print results of the model to the screen and the log file:
        print_and_log("Model: %s" % (model_type))    
        print_and_log("ACC [Train, Val, Test]: {}".format(tab_acc[model_index]))
        print_and_log("TPR [Train, Val, Test]: {}".format(tab_tpr[model_index]))
        print_and_log("PPV [Train, Val, Test]: {}".format(tab_ppv[model_index]))
        print_and_log("F1  [Train, Val, Test]: {}\n".format(tab_f1[model_index]))
           
        # Save the results of the model in CSV files
        np.savetxt(acc_file, tab_acc, delimiter=',', fmt='%0.4f')
        np.savetxt(tpr_file, tab_tpr, delimiter=',', fmt='%0.4f')
        np.savetxt(ppv_file, tab_ppv, delimiter=',', fmt='%0.4f')
        np.savetxt(f1_file,  tab_f1,  delimiter=',', fmt='%0.4f')
     

# Ensembles of the best models (one instance of each):
def ensembles():

    # Ground-truth of the test subset    
    df_test = load_data(LABELS_TEST_FILE)
    y_true = df_test["category"].replace({'positive':1, 'negative':0})
    
    # Matrix to save the average scores and standard deviation
    avg_scores = []
    std_scores = []
    
    print_and_log("Ensembles of one instance of each of the Top N models:\n")
    
    # Load the F1 scores
    f1_file = "../results/covid-tl-f1-"  + str(POOLING) + ".csv"
    f1_scores = np.loadtxt(f1_file, delimiter=',')
    
    # Gets the list of best performing methods regarding F1 score
    # from the worst to the best.
    f1_ranking = np.argsort(f1_scores[:,2])
        
    # List of ensembles to construct using the top-N models
    # Top-2, Top-3, ..., Top-7, Top-21
    ens_list = np.concatenate((np.arange(2,8),[len(MODEL_TYPE_LIST)]))
    
    
    # For each list of Top N models:
    for ens in ens_list:              
        scores = []
        predictions = np.empty([ens, 400, 2])
        # Get the top 'ens' best performers.
        ens_members = np.take(MODEL_TYPE_LIST, f1_ranking[-ens:])         
        for rep in range(0,N_REPEATS):          
            # Load predictions
            for index, model_type in enumerate(ens_members):
                pred_filename = "../results/predictions/covid-tl-pred-" + model_type + "-" + POOLING + "-" + str(rep+1) + ".csv"
                predictions[index] = np.loadtxt(pred_filename, delimiter=',')
            # Average the predictions
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
            
            scores.append([acc, tpr, ppv, f1])
                        
            #print("Repetition: %i of %i" % (rep+1, N_REPEATS))
            #print("Acc: %.4f TPR: %.4f PPV: %.4f F1: %.4f\n" % (acc, tpr, ppv, f1))    
        
        m_acc, m_tpr, m_ppv, m_f1 = np.mean(scores,0)
        s_acc, s_tpr, s_ppv, s_f1 = np.std(scores,0)
        
        avg_scores.append([m_acc, m_tpr, m_ppv, m_f1])
        std_scores.append([s_acc, s_tpr, s_ppv, s_f1])

        print_and_log("Ensemble of the Top %i F1 performers:" % ens)
        print_and_log("{}".format(ens_members))
        print_and_log("Mean Acc: %.4f (+/- %.4f)" % (m_acc, s_acc))
        print_and_log("Mean TPR: %.4f (+/- %.4f)" % (m_tpr, s_tpr))
        print_and_log("Mean PPV: %.4f (+/- %.4f)" % (m_ppv, s_ppv))
        print_and_log("Mean F1:  %.4f (+/- %.4f)\n" % (m_f1, s_f1))                   
        
    # Save the average scores to a CSV file. Each line corresponds to a model.
    # Columns correspond to ACC, TPR, PPV, and F1, respectively.
    avg_scores_filename = '../results/ensembles/covid-tl-ensemble-avg.csv'
    std_scores_filename = '../results/ensembles/covid-tl-ensemble-std.csv'
    os.makedirs(os.path.dirname(avg_scores_filename), exist_ok=True)
    os.makedirs(os.path.dirname(std_scores_filename), exist_ok=True)
    np.savetxt(avg_scores_filename, avg_scores, delimiter=',', fmt='%0.4f')           
    np.savetxt(std_scores_filename, std_scores, delimiter=',', fmt='%0.4f')

# Ensembles of instances of the each model invididually
def ensembles_single_model():

    # matrix to hold thepredictions
    predictions = np.empty([N_REPEATS, 400, 2])
    # matrix to save the scores
    scores = []

    # ground-truth of the test subset    
    df_test = load_data(LABELS_TEST_FILE)
    y_true = df_test["category"].replace({'positive':1, 'negative':0})

    print_and_log("Ensembles of %i instances of the same model:\n" % N_REPEATS)
    
    # For each model
    for model_index, model_type in enumerate(MODEL_TYPE_LIST):
        # For each repetition of the same model
        for rep in range(0,N_REPEATS):
            # Load the predictions
            pred_filename = "../results/predictions/covid-tl-pred-" + model_type + "-" + POOLING + "-" + str(rep+1) + ".csv"
            predictions[rep] = np.loadtxt(pred_filename, delimiter=',')
            
        # Get the average predictions
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
        
        scores.append([acc, tpr, ppv, f1])
        
        # print results to screen   
        print_and_log("Model: %s" % (model_type))
        print_and_log("Acc: %.4f TPR: %.4f PPV: %.4f F1: %.4f\n" % (acc, tpr, ppv, f1))                        

    # Save the scores to a CSV file. Each line corresponds to a model.
    # Columns correspond to ACC, TPR, PPV, and F1, respectively.
    scores_filename = '../results/ensembles/covid-tl-ensemble-1model.csv'
    os.makedirs(os.path.dirname(scores_filename), exist_ok=True)
    np.savetxt(scores_filename, scores, delimiter=',', fmt='%0.4f')    

def ensembles_all_instances():

    # Ground-truth of the test subset    
    df_test = load_data(LABELS_TEST_FILE)
    y_true = df_test["category"].replace({'positive':1, 'negative':0})
    
    # Matrix to save the scores
    scores = []
    
    print_and_log("Ensembles of all instances of each of the Top N models:\n")
    
    # Load the F1 scores
    f1_file = "../results/covid-tl-f1-"  + str(POOLING) + ".csv"
    f1_scores = np.loadtxt(f1_file, delimiter=',')
    
    # Gets the list of best performing methods regarding F1 score
    # from the worst to the best.
    f1_ranking = np.argsort(f1_scores[:,2])
        
    # List of ensembles to construct using the top-N models:
    ens_list = np.concatenate((np.arange(2,8),[len(MODEL_TYPE_LIST)]))

    # For each list of Top N models:
    for ens in ens_list:              
        scores = []
        predictions = np.empty([ens*N_REPEATS, 400, 2])      
        # Get the top 'ens' best performers.
        ens_members = np.take(MODEL_TYPE_LIST, f1_ranking[-ens:])            
        for index, model_type in enumerate(ens_members):
            for rep in range(0,N_REPEATS):
                # Load predictions           
                pred_filename = "../results/predictions/covid-tl-pred-" + model_type + "-" + POOLING + "-" + str(rep+1) + ".csv"
                predictions[index*N_REPEATS + rep] = np.loadtxt(pred_filename, delimiter=',')
        # Average the predictions
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
           
        scores.append([acc, tpr, ppv, f1])
                       
        print_and_log("Ensemble of the Top %i F1 performers:" % ens)
        print_and_log("{}".format(ens_members))
        print_and_log("Acc: %.4f TPR: %.4f PPV: %.4f F1: %.4f\n" % (acc, tpr, ppv, f1))          
             
       
        # Save the average scores to a CSV file. Each line corresponds to a model.
        # Columns correspond to ACC, TPR, PPV, and F1, respectively.
        scores_filename = '../results/ensembles/covid-tl-ensemble-all-instances.csv'
        os.makedirs(os.path.dirname(scores_filename), exist_ok=True)
        np.savetxt(scores_filename, scores, delimiter=',', fmt='%0.4f')           
        
# Main
from time import perf_counter
t1_start = perf_counter()
    
# Get hostname for log-files
import socket
hostname = socket.gethostname()

# Create log filename 
# Includes the hostname to avoid conflicts when running in multiple machines
# with synced content. 
log_filename = "../results/main" + "-" + hostname + ".log"

# Logfile header
print_and_log("Machine: %s" % hostname)
from datetime import datetime
now = datetime.now()
print_and_log(now.strftime("Date: %d/%m/%Y Time: %H:%M:%S\n"))

# Train (if retrain is True) and test all models
train_test_all_models()  
# Get ensembles of the top models (one instance of each):
ensembles()    
# Get ensembles of instances of each model individually:
ensembles_single_model()
# Get ensembles of all instances of each model individually:
ensembles_all_instances()

t1_stop = perf_counter()
t1_elapsed = t1_stop-t1_start
print_and_log("Elapsed time: %0.4f seconds\n" % t1_elapsed)