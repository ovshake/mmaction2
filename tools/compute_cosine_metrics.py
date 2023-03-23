import pickle
import matplotlib.pyplot as plt
import numpy as np
for domain in ['speed', 'color', 'vcop']:
    for split in ['D1', 'D2', 'D3']:
        with open(f'/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/{domain}/{split}/output_eval.pkl', 'rb') as f:
            data = pickle.load(f)
        norm_array = []
        for i in range(len(data)):
            data[i] = np.linalg.norm(data[i])
            norm_array.append(data[i])
        plt.hist(norm_array, bins=100)
        plt.title(f'{domain} {split} histogram')
        plt.savefig(f'/data/abhishek/projects/mmaction2/work_dirs/cosine_eval_features/{domain}/{split}/output_eval.png')
        plt.xlabel(xlabel='l2 Norm')
        plt.ylabel(ylabel='Frequency')
        plt.clf()


