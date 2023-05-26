import sys
import torch
import os
from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from theconf import Config as C


def get_model(conf, num_class=10, data_parallel=True):
    name = conf['model']['type']

    if 'BertForSequenceClassification' in name:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_class)
    else:
        raise NameError('no model named, %s' % name)
    
    if torch.cuda.is_available():
        if data_parallel:
            model = model.cuda()
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
            model = DataParallel(model)
        else:
            import horovod.torch as hvd
            device = torch.device('cuda', hvd.local_rank())
            model = model.to(device)
        cudnn.benchmark = True
    return model


def get_num_class(dataset):
    return C.get()['num_classes']

