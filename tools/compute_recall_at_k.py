import pickle
import pandas as pd
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def read_annotation_file(file_path):
    df = pd.read_pickle(file_path)
    return df

def parse_args():
    parser = argparse.ArgumentParser(description='Compute recall@k')
    parser.add_argument(
        '--annotation-file-domain-a',
        help='annotation file path',
        required=True)

    parser.add_argument(
        '--annotation-file-domain-b',
        help='annotation file path',
        required=True)

    parser.add_argument(
        '--output_file_domain_a',
        help='output file path',
       required=True)
    parser.add_argument(
        '--output_file_domain_b',
        help='output file path',
        required=True
    )
    parser.add_argument(
        '--top-k', type=int, default=5, help='top k predictions')
    args = parser.parse_args()
    return args

# Write a function to load file from pickle file
def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# write a function to compute vector_a's cosine similarity with
# all vectors in matrix matrix_b
def compute_cosine_similarity(vector_a, matrix_b):
    similarity = np.dot(vector_a, matrix_b.T) / (
        np.linalg.norm(vector_a) * np.linalg.norm(matrix_b, axis=1))
    return similarity


# write a function to compute cosine similarity
# matrix from a row of vectors
def compute_recall_at_k(ann_info_a,
                        ann_info_b,
                        output_pkl_a,
                        output_pkl_b,
                        top_k=5):

    # similarity_matrix = cosine_similarity(output_pkl)
    tp = 0
    for index_a, row_a in ann_info_a.iterrows():
        feature_a = output_pkl_a[index_a]
        label_a = row_a['verb_class']
        similarity = compute_cosine_similarity(feature_a, output_pkl_b)
        similarity[index] = -2
        top_k_similarities = similarity.argsort()[-top_k:][::-1]
        top_k_labels = ann_info_b.iloc[top_k_similarities]['verb_class']
        if label in top_k_labels:
            tp += 1
    return tp / len(ann_info_a)



args = parse_args()
output_pkl_a = load_pickle_file(args.output_file_domain_a)
output_pkl_b = load_pickle_file(args.output_file_domain_b)
ann_info_domain_a = read_annotation_file(args.annotation_file_domain_a)
ann_info_domain_b = read_annotation_file(args.annotation_file_domain_b)
recall_at_k = compute_recall_at_k(ann_info, output_pkl)
print(f"Recall at {args.top_k} is {recall_at_k}")