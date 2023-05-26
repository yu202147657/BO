from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import torchtext
import csv
import os
import numpy as np
import random
import pandas as pd
import re
from theconf import Config as C
from ..common import get_logger
import logging
from datasets import Dataset

logger = get_logger('Text AutoAugment')
logger.setLevel(logging.INFO)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        

def clean_web_text(st):
    """clean text."""
    st = st.replace("<br />", " ")
    st = st.replace("&quot;", "\"")
    st = st.replace("<p>", " ")
    if "<a href=" in st:
        # print("before:\n", st)
        while "<a href=" in st:
            start_pos = st.find("<a href=")
            end_pos = st.find(">", start_pos)
            if end_pos != -1:
                st = st[:start_pos] + st[end_pos + 1:]
            else:
                print("incomplete href")
                print("before", st)
                st = st[:start_pos] + st[start_pos + len("<a href=")]
                print("after", st)

        st = st.replace("</a>", "")
        # print("after\n", st)
        # print("")
    st = st.replace("\\n", " ")
    st = st.replace("\\", " ")
    # while "  " in st:
    #   st = st.replace("  ", " ")
    return st


def get_examples(dataset, text_key='text'):
    """get dataset examples"""
    examples = []
    for i in range(dataset.num_rows):
        guid = i
        text_a = dataset[i][text_key]
        label = dataset[i]['label']
        text_a = clean_web_text(text_a)
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def general_split(examples, test_size, train_size, n_splits=2, split_idx=0):
    """used for datasets on huggingface"""
    labels = [e.label for e in examples]
    kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                    random_state=C.get()['seed'])
    kf = kf.split(list(range(len(examples))), labels)
    for _ in range(split_idx + 1):  # split_idx equal to cv_fold. this loop is used to get i-th fold
        train_idx, valid_idx = next(kf)
    train_dev_set = np.array(examples)
    return list(train_dev_set[train_idx]), list(train_dev_set[valid_idx])


def rebalance_resize(train_dataset, class_imbalance, sparsity, seed):
    #Create the desired sparsity and imbalance (binary class). 
    # data is HF dataset object
    label_1_examples = train_dataset.filter(lambda example: example['label'] == 1).shuffle(seed=seed)
    label_0_examples = train_dataset.filter(lambda example: example['label'] == 0).shuffle(seed=seed)

    # Compute the number of examples to sample from each label
    num_label_1_examples = int(class_imbalance * sparsity)
    num_label_0_examples = int((1 - class_imbalance) * sparsity)
    
    # Sample examples from each label and concatenate them
    dict_label_1 = label_1_examples[:num_label_1_examples]
    dict_label_0 = label_0_examples[:num_label_0_examples]
    
    #convert to HF dataset
    combined_dict = {k: dict_label_1[k] + dict_label_0[k] for k in dict_label_1}

    dataset = Dataset.from_dict(combined_dict)
    return dataset



class Wikipedia(Dataset):
    def __init__(self):
        self.label_list = list(range(2000, 2022+1))
    
    def get_label_map(self):
        
        label_map = {}
        for label in self.label_list:
            if label >= 2010:
                label_map[label] = 1
            else:
                label_map[label] = 0
                
        return label_map

class Amazon(Dataset):
    def __init__(self):
        self.label_list = [1, 2]
    
    def get_label_map(self):
        
        label_map = {}
        for label in self.label_list:
            if label == 2:
                label_map[label] = 1
            else:
                label_map[label] = 0
                
        return label_map


def get_processor(dataset_name):
    """get processor."""
    dataset_name = dataset_name.lower()
    processors = {
        "wiki": Wikipedia,
        "amazon": Amazon
    }
    processor = processors[dataset_name]()
    return processor


def refactor_classes(dataset, dataset_name, num_classes):
    """Take dataset and desired number of classes
    Refactor current number of classes to desired num (num_classes)
    """
    pc = get_processor(dataset_name)

    label_map = pc.get_label_map()
    
    refactored_dataset = dataset.map(lambda example: {'label': label_map[example['label']]})
    
    return refactored_dataset


if __name__ == '__main__':
    from datasets import load_dataset
    
    pc = get_processor('wiki')
    data_files = {'train': 'clean_sentences_small_wiki_train.csv', 'test': 'clean_sentences_small_wiki_train.csv'}
    dataset = load_dataset('csv', data_dir='/home/jovyan/examples/examples/pytorch/MPhil_BO/taa/Data', data_files=data_files, split='train')
    
    label_map = pc.get_label_map()
    
    refactored_dataset = dataset.map(lambda example: {'label': label_map[example['label']]})
    
    print(set(list(refactored_dataset['label'])))
