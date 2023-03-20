import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_image_data(image_dir,test_size = 0.1):
    
    images = []
    label = []
    classes = {}

    for idx, folder in enumerate(os.listdir(image_dir)):
        image_path = os.path.join(image_dir,folder)
        target = folder
        classes[target] = idx
        for image in os.listdir(image_path):
            img = Image.open(os.path.join(image_path, image))
            resized_img = img.resize((128,128))
            normalise_img = np.array(resized_img) / 255
            images.append(normalise_img)
            
            label.append(target)
            
    images = np.array(images)
    
    for i in range(len(label)):
        label[i] = classes[label[i]]
        
    label = np.array(label)
    
    x_train, x_test, y_train, y_test = train_test_split(images, label,test_size= test_size, shuffle= True)
    
    # Flatten image data
    x_train = x_train.reshape(x_train.shape[0],-1).T
    x_test = x_test.reshape(x_test.shape[0],-1).T
    
    y_train = y_train.reshape(y_train.shape[0],1).T
    y_test = y_test.reshape(y_test.shape[0],1).T
            
    return x_train, x_test, y_train, y_test