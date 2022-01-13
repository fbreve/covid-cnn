# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:17:15 2020

@author: fbrev

Extract features from the COVID-Net dataset train examples using VGG16 and
VGG19 and saves them to text files.

Required packages:
    - numpy
    - pandas
    - tensorflow / tensorflow-gpu
    - pillow
"""

# Comment these two lines to use GPU. I added them because ResNet and Xception
# causes a OOM error on my GPU (RTX 2060 SUPER), so I had to run them on CPU.
# VGG runs on GPU just fine.
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # don't use GPUs


import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#print(os.listdir("../via-dataset/images/"))

IMAGE_CHANNELS=3

POOLING = 'avg' # None, 'avg', 'max'

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
    image_size = (224, 224)
    if model_type=='VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
    elif model_type=='VGG19':
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        model = VGG19(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
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
    elif model_type=='Xception':
        image_size = (299, 299)
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        model = Xception(weights='imagenet', include_top=False, pooling=POOLING, input_shape=image_size + (IMAGE_CHANNELS,))
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
    
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model

    output = Flatten()(model.layers[-1].output)   
    model = Model(inputs=model.inputs, outputs=output)

    return model, preprocessing_function, image_size

def extract_features(df, model, preprocessing_function, image_size, dataset_path):

    df["category"] = df["category"].replace({1: 'positive', 0: 'negative'}) 
           
    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )
    
    total = df.shape[0]
    batch_size = 4
    
    generator = datagen.flow_from_dataframe(
        df, 
        dataset_path, 
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    features = model.predict(generator, steps=np.ceil(total/batch_size))
    
    return features
   
# Main
    
#model_type_list = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152',
#                   'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'Xception',
#                   'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 
#                   'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
#                   'EfficientNetB6', 'EfficientNetB7', ]
model_type_list = ['EfficientNetB2']

df_train = load_data(LABELS_TRAIN_FILE)
np.savetxt("covid-train-labels.txt", df_train.category, fmt="%s")

df_test = load_data(LABELS_TEST_FILE)
np.savetxt("covid-test-labels.txt", df_test.category, fmt="%s")
   
for model_type in model_type_list:    
    
      model, preprocessing_function, image_size = create_model(model_type)

      features = extract_features(df_train, model, preprocessing_function, image_size, DATASET_TRAIN_PATH)
      filename = "covid-train-" + model_type + "-data.txt"
      np.savetxt(filename, features, fmt="%s")
      
      features = extract_features(df_test, model, preprocessing_function, image_size, DATASET_TEST_PATH)
      filename = "covid-test-" + model_type + "-data.txt"
      np.savetxt(filename, features, fmt="%s")
      