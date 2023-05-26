import sys
from theconf import Config as C
from .utils.eda import gen_eda
from .utils.backtranslation import gen_back_translate
import os
import ray
import torch
import random
import numpy as np
import pickle


def full(texts, labels, mode_fn, alpha, n_aug):
    #filter the text to augment
    texts = texts
    
    #call aug_mode function (eg backtranslate)
    if mode_fn.__name__== "eda":
        args = [texts, labels, alpha, n_aug]
    else:
        args = [texts, labels]
    
    aug_texts, aug_labels = mode_fn(*args)
    return aug_texts, aug_labels


def full_minority(texts, labels, mode_fn, alpha, n_aug):
    #filter the text to augment
    idx = [i for i, x in enumerate(labels) if x == 0]
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    
    #call aug_mode function (eg backtranslate)
    if mode_fn.__name__== "eda":
        args = [texts, labels, alpha, n_aug]
    else:
        args = [texts, labels]
    
    aug_texts, aug_labels = mode_fn(*args)
    return aug_texts, aug_labels


def random_aug(texts, labels, mode_fn, alpha, n_aug, n=0.25):
    """Randomly Select n% of samples to augment"""
    
    idx = random.sample(range(len(texts)), int(n*len(texts)))
    
    aug_texts = [texts[i] for i in idx]
    aug_labels = [labels[i] for i in idx]
    
    #call aug_mode function (eg backtranslate)
    if mode_fn.__name__== "eda":
        args = [aug_texts, aug_labels, alpha, n_aug]
    else:
        args = [aug_texts, aug_labels]
    
    aug_texts, aug_labels = mode_fn(*args)
    
    #combine aug and original texts
    comb_texts = texts + aug_texts
    comb_labels = labels + aug_labels
    
    return comb_texts, comb_labels

def influential_aug(texts, labels, mode_fn, alpha, n_aug, n=0.25):
    """Augment the samples with the top 25% highest loss"""
    C.get()
    abspath = C.get()["abspath"]
    sparsity = C.get()["sparsity"]
    path = os.path.join(abspath, f'mlruns/losses_{sparsity}.pkl')
    with open(path, "rb") as f:
        losses = pickle.load(f)
        
    # Calculate the threshold
    threshold = int(len(texts) * 0.25)
    # Calculate the indices of the top 25% numbers
    indices = np.argsort(losses)[-threshold:]

    # Use the indices to index the other list
    aug_texts = [texts[i] for i in indices]
    aug_labels = [labels[i] for i in indices]
    
    #call aug_mode function (eg backtranslate)
    if mode_fn.__name__== "eda":
        args = [aug_texts, aug_labels, alpha, n_aug]
    else:
        args = [aug_texts, aug_labels]
    
    aug_texts, aug_labels = mode_fn(*args)
    
    #combine aug and original texts
    comb_texts = texts + aug_texts
    comb_labels = labels + aug_labels
    
    return comb_texts, comb_labels


def influential_minority_aug(texts, labels, mode_fn, alpha, n_aug, n=0.25):
    """Augment the samples with the top 25% highest loss that belong
    to the minority class"""
    C.get()
    abspath = C.get()["abspath"]
    sparsity = C.get()["sparsity"]
    path = os.path.join(abspath, f'mlruns/losses_{sparsity}.pkl')
    with open(path, "rb") as f:
        losses = pickle.load(f)
        
    # Calculate the threshold
    threshold = int(len(texts) * 0.25)
    # Calculate the indices of the top 25% numbers
    indices = np.argsort(losses)[-threshold:]

    # Use the indices to get the influential points
    aug_texts = [texts[i] for i in indices]
    aug_labels = [labels[i] for i in indices]
    #filter for minority class label
    idx = [i for i, x in enumerate(aug_labels) if x == 0]
    aug_texts = [aug_texts[i] for i in idx]
    aug_labels = [aug_labels[i] for i in idx]
    
    #call aug_mode function (eg backtranslate)
    if mode_fn.__name__== "eda":
        args = [aug_texts, aug_labels, alpha, n_aug]
    else:
        args = [aug_texts, aug_labels]
        
    aug_texts, aug_labels = mode_fn(*args)
    
    #combine aug and original texts
    comb_texts = texts + aug_texts
    comb_labels = labels + aug_labels
    
    return comb_texts, comb_labels


def random_forget(texts, labels, mode_fn, alpha, n_aug, n=0.25):
    """Randomly Select n% of samples to forget"""
    indices_to_delete = random.sample(range(len(texts)), int(n*len(texts)))

    # Delete the selected elements from texts and labels
    texts = [text for i, text in enumerate(texts) if i not in indices_to_delete]
    labels = [label for i, label in enumerate(labels) if i not in indices_to_delete]

    return texts, labels

def random_forget_majority(texts, labels, mode_fn, alpha, n_aug, n=0.25):
    """Randomly Select n% of samples to forget from the majority class"""
    majority_idx = [i for i, x in enumerate(labels) if x == 1]
    indices_to_delete = random.sample(majority_idx, int(len(majority_idx) * 0.25))

    #Delete the selected elements from texts and labels
    texts = [text for i, text in enumerate(texts) if i not in indices_to_delete]
    labels = [label for i, label in enumerate(labels) if i not in indices_to_delete]

    return texts, labels
    

def forget_majority(texts, labels, mode_fn, alpha, n_aug):
    """Delete all examples of majority class
    Meaning - keep only minority class label (0)"""
    
    idx = [i for i, x in enumerate(labels) if x == 0]
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    
    return texts, labels


def influential_majority_forget(texts, labels, mode_fn, alpha, n_aug, n=0.25):  
    """Forget the influential samples that belong
    to the majority class"""
    C.get()
    abspath = C.get()["abspath"]
    sparsity = C.get()["sparsity"]
    path = os.path.join(abspath, f'mlruns/losses_{sparsity}.pkl')
    with open(path, "rb") as f:
        losses = pickle.load(f)

    # Calculate the threshold
    threshold = int(len(texts) * 0.25)
    # Calculate the indices of the top 25% numbers
    influential_idx = np.argsort(losses)[-threshold:]
    #indices for majority class
    majority_idx = [i for i, x in enumerate(labels) if x == 1]
    #influential AND majority samples
    indices_to_delete = set(influential_idx).intersection(set(majority_idx))

    texts = [text for i, text in enumerate(texts) if i not in indices_to_delete]
    labels = [label for i, label in enumerate(labels) if i not in indices_to_delete]
    
    return texts, labels


def duplicate(texts, labels, mode_fn, alpha, n_aug):
    """Repeating the text n_aug times"""
    aug_texts = [item for item in texts for i in range(n_aug)]
    aug_labels = [item for item in labels for i in range(n_aug)]
    return aug_texts, aug_labels

def backtranslation(texts, labels):
    aug_texts = gen_back_translate(texts, batch_size=64)
    return aug_texts, labels

def eda(texts, labels, alpha, n_aug):
    aug_texts, aug_labels = gen_eda(texts, labels, alpha=alpha, num_aug=n_aug)
    return aug_texts, aug_labels

def mode_list():  
    l = [backtranslation, eda]
    return l

def method_list():
    l = [full, full_minority, random_aug, influential_aug, influential_minority_aug, random_forget, random_forget_majority, forget_majority, influential_majority_forget]
    return l

def augment_list(alpha, n_aug):  
    
    aug_mode = mode_list()
    aug_method = method_list()
    
    l = []
    
    for augment_fn in aug_method:
        for mode_fn in aug_mode:
            l.append((augment_fn, mode_fn, alpha, n_aug))
    return l



def get_augment(name, alpha, n_aug):
    """Create a dict where the key is function string, 
    and value is the function"""
    augment_dict = {f"{augment_fn.__name__}_{mode_fn.__name__}": (augment_fn, mode_fn, v1, v2) for augment_fn, mode_fn, v1, v2 in augment_list(alpha, n_aug)}
    #AUGMENT DICT example {'full_eda': (<function full at 0x7f2b3a2ce700>, <function eda at 0x7f2b37c6f1f0>, 0.05, 8)}
    return augment_dict[name]


# @ray.remote
# def apply_augment(text, sub_policy, config):
def apply_augment(policy, config, texts, labels):
    C.get()
    C.get().conf = config
    #only need the below line if topN =3
    #sub_policy = random.choice(policy)  
    #method, mode, pr = sub_policy[0]
    method, mode, pr = policy[0][0]
    if pr >= 0.5:
        name = f"{method}_{mode}"
        augment_fn, mode_fn, alpha, n_aug = get_augment(name, C.get()['alpha'], C.get()['n_aug'])
        texts, labels = augment_fn(texts, labels, mode_fn, alpha, n_aug)    

    return texts, labels



if __name__ == '__main__':
    l = augment_list(0.05, 8)
    print(l)