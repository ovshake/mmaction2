import numpy as np 
import pickle
import os
import pandas as pd
import random
x = "/data/jongmin/projects/SADA_Domain_Adaptation_Splits/D1_train.pkl"
df = pd.read_pickle(x)
print(df)
q = 0
for _, line in df.iterrows():

    start_frame = int(line['start_frame'])
    end_frame = int(line['stop_frame'])
    all_frame = end_frame - start_frame +1
    # print(all_frame)
    if all_frame < 32 :
        print(all_frame)
        q += 1
print(q)
random.seed(42)
x = random.randint(1, 4)
print(x)