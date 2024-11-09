import pickle

import numpy as np

def process(scaled_coef):
    with open("E:/andan/Flask_project_integrate-ml/models/lr_model.pkl", "rb") as f:
        regressor = pickle.load(f)
    predicted_mu = regressor.predict(scaled_coef) 
    

    with open("E:/andan/Flask_project_integrate-ml/models/scaler_y.pkl", "rb") as f:
        min_max_scaler_y = pickle.load(f)
    mu = np.round(min_max_scaler_y.inverse_transform(predicted_mu.reshape(1, -1)), 2)
        
    return mu[0][0]


def preprocess(coef):
    with open("E:/andan/Flask_project_integrate-ml/models/scaler_x.pkl", "rb") as f:
        min_max_scaler_x = pickle.load(f)

    scaled_coef = min_max_scaler_x.transform ([[coef]])
    return scaled_coef

