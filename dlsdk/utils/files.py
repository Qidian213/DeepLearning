import os
import json
import pickle
import numpy as np
import pandas as pd


def make_dir(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    return dst_dir


def read_txt(file):
    context = []
    with open(file, 'r') as f:
        cts = f.readlines()
        for c in cts:
            context.append(c.strip())
    return context


def save_txt(file, context, operator='w'):
    with open(file, operator) as f:
        for c in context:
            f.write(c.strip() + '\n')
        f.close()


def read_pickle(file):
    with open(file, 'rb') as fp:
        context = pickle.load(fp)
    return context


def save_pickle(file, context):
    with open(file, 'wb') as fp:
        pickle.dump(context, fp)


def read_json(file):
    with open(file, 'r') as f:
        context = json.load(f)
    return context


def save_json(file, context, indent=None):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=indent, ensure_ascii=False)


def read_csv(file):
    csv_data = pd.read_csv(file)
    csv_data = pd.DataFrame(csv_data) # columns=['BI', 'CW', 'CX', 'BT', 'BY']
    csv_data = csv_data.replace(np.nan, None)
    return csv_data
