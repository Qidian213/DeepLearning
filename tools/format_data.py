
import os
import json
import pickle
import random
import pandas as pd


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

def read_json(file):
    with open(file, 'r') as f:
        context = json.load(f)
    return context

def save_json(file, context, indent=None):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=indent, ensure_ascii=False)

translate = {
    "cane": "dog", 
    "dog": "dog", 
    "cavallo": "horse", 
    "elefante": "elephant", 
    "farfalla": "butterfly",
    "gallina": "chicken", 
    "gatto": "cat", 
    "mucca": "cow", 
    "pecora": "sheep", 
    "ragno": "spider", 
    "scoiattolo": "squirrel"
}

### 从isssue 拉取数据
if __name__ == "__main__":

    src_dir = "./data/Animals-10"

    class_names = os.listdir(src_dir)

    train_list = []
    eval_list = []

    for class_name in class_names:
        class_dir = os.path.join(src_dir, class_name)
        class_files = os.listdir(class_dir)

        for class_file in class_files:
            class_file_path = os.path.join(class_dir, class_file)

            if random.random() >0.2:
                train_list.append({
                    "file_path": class_file_path,
                    "class_name": translate[class_name]
                })
            else:
                eval_list.append({
                    "file_path": class_file_path,
                    "class_name": translate[class_name]
                })
        save_json("./data/Animals_Train.json", train_list, indent=4)
        save_json("./data/Animals_Eval.json", eval_list, indent=4)