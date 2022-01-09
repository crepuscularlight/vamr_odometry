import numpy as np
import pandas as pd

def load_K(data_dir):
    data=pd.read_csv(data_dir+"K.txt",header=None)
    K=data.iloc[0:3,0:3]
    K=K.to_numpy()

    return K

def load_dataset(dataset):
    pass

def image_name(dataset):
    pass
