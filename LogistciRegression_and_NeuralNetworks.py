import numpy as np
import os
from prepare_image_data import prepare_image_data 
from LogisticRegression_class import LogisticRegression

if __name__ == '__main__':
    working_dir = os.path.dirname(os.path.realpath(__file__))
    image_dir = os.path.join(working_dir,"data/images/image_classification")
    
    x_train, x_test, y_train, y_test = prepare_image_data(image_dir)
    logisticregression_model = LogisticRegression(x_train, y_train, num_iterations= 200, lr = 0.001)
    test_prediction = logisticregression_model.predict(x_test, y_test)
    
