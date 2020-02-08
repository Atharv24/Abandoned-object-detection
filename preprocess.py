import numpy as np
import pandas as pd


label_dict = pd.read_csv('label.csv')


for i in range(11):
    frames = np.load(f'data/tensor{i}.npy')
    label = np.zeros((frames.shape[0]))
    if i!=5 and i!=10:
        start = int(label_dict['start'][i])
        label[start:] = 1
    elif i==5:
        start = int(label_dict['start'][i])
        end_ = int(label_dict['end_'][i])
        start1 = int(label_dict['start1'][i])
        end1 = int(label_dict['end1'][i])
        label[start:end_] = 1
        label[start1:end1] = 1
    
    np.save(f'label{i}.npy', label)

