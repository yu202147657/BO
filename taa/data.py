import sys
import logging
import os
import random
import torch
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from theconf import Config as C
import numpy as np
from .custom_dataset import general_dataset
from datasets import load_dataset
from .augmentation_method import apply_augment
from .common import get_logger
import pandas as pd
from .utils.raw_data_utils import get_examples, general_split, rebalance_resize, refactor_classes
from transformers import BertTokenizer, BertTokenizerFast
from .text_networks import get_num_class
import math
import copy
#from .archive import policy_map
from functools import partial
import time
from .utils.get_data import download_data
from .utils.metrics import n_dist
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
import multiprocessing 
# Global variable to track whether multiprocessing has been initialized
multiprocessing_initialized = False

def init_multiprocessing():
    global multiprocessing_initialized
    if not multiprocessing_initialized:
        multiprocessing.set_start_method('spawn')
        multiprocessing_initialized = True
#init_multiprocessing()

logger = get_logger('Text AutoAugment')
logger.setLevel(logging.INFO)

def get_datasets(dataset, policy_opt):
    # do data augmentation
    aug = C.get()['aug']
    path = C.get()['dataset']['path']
    data_dir = C.get()['dataset']['data_dir']
    data_files = C.get()['dataset']['data_files']
    data_name = C.get()['dataset']['name']
    text_key = C.get()['dataset']['text_key']
    sparsity = C.get()['sparsity']
    class_imbalance = C.get()['class_imbalance']
    num_classes = C.get()['num_classes']
    seed = C.get()['seed']

    # load dataset
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if path is None:
        train_dataset = load_dataset(path=dataset, data_dir=data_dir, data_files=data_files, split='train')
        test_dataset = load_dataset(path=dataset, data_dir=data_dir, data_files=data_files, split='test')
    elif path=='custom': #custom dataset
        train_dataset = load_dataset('csv', data_dir=data_dir, data_files=data_files, split='train')
        test_dataset = load_dataset('csv', data_dir=data_dir, data_files=data_files, split='test')
    else: #glue datasets 
        train_dataset = load_dataset(path=path, name=dataset, data_dir=data_dir, data_files=data_files, split='train')
        test_dataset = load_dataset(path=path, name=dataset, data_dir=data_dir, data_files=data_files, split='test')
    
    #Relabel classes
    try:
        train_dataset = refactor_classes(train_dataset, data_name, num_classes)
        test_dataset = refactor_classes(test_dataset, data_name, num_classes)
    except: #doesnt need refactored classes
        print('Not refactoring classes')
    
    #Introduce class imbalance and sparsity
    train_dataset = rebalance_resize(train_dataset, class_imbalance, sparsity, seed)
    #Make test set same size as eval. RETURN to this assumption 
    test_dataset = rebalance_resize(test_dataset, class_imbalance, int(0.2 * sparsity), seed)
    
    all_train_examples = get_examples(train_dataset, text_key)

    train_examples, valid_examples = general_split(all_train_examples, int(0.2 * sparsity),
                                                   int(0.8 * sparsity))
    test_examples = get_examples(test_dataset, text_key)

    #Augmentation ONLY happens on train dataset
    logger.info('aug: {}'.format(aug))
    if isinstance(aug, list) or aug in ['taa', 'random_taa']: 
        print(aug, 'AUG METHOD #######################')
        transform_train = Augmentation(aug)
        train_dataset = general_dataset(train_examples, tokenizer, text_transform=transform_train)
    else:
        train_dataset = general_dataset(train_examples, tokenizer, text_transform=None)
        
    val_dataset = general_dataset(valid_examples, tokenizer, text_transform=None)
    test_dataset = general_dataset(test_examples, tokenizer, text_transform=None)
    logger.info('len(trainset) = %d; len(validset) = %d; len(testset) = %d' %
                (len(train_dataset), len(val_dataset), len(test_dataset)))
    return train_dataset, val_dataset, test_dataset



class Augmentation(object):
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, texts, labels):
        texts, labels = apply_augment(policy=self.policy, config=copy.deepcopy(C.get().conf), texts=texts, labels=labels)
        
        n_dist_value = n_dist(texts, 2)  # ngram=2
        
        return texts, labels, n_dist_value




def augment(dataset, policy, class_imbalance, sparsity, n_aug, configfile=None):
    # get searched augmentation policy
    #   _ = C('confs/%s' % configfile)
    C.get()['n_aug'] = n_aug
    text_key = C.get()['dataset']['text_key']

    transform_train = Augmentation(policy)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    examples = get_examples(dataset, text_key)

    augmented_dataset = general_dataset(examples, tokenizer, text_transform=transform_train)    

    return augmented_dataset
