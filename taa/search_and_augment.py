import pkg_resources
from .data import augment
from .search import search_policy
from datasets import load_dataset
from theconf import Config as C
#from .archive import policy_map
from .utils.raw_data_utils import refactor_classes, rebalance_resize


def search_and_augment(configfile=None):
    """search optimal policy and then use it to augment texts"""
    if configfile is None:
        resource_package = __name__
        resource_path = '/'.join(('confs', 'bert_sst2_example.yaml'))
        configfile = pkg_resources.resource_filename(resource_package, resource_path)

    _ = C(configfile)

    name = C.get()['dataset']['name']
    path = C.get()['dataset']['path']
    data_dir = C.get()['dataset']['data_dir']
    data_files = C.get()['dataset']['data_files']
    data_name = C.get()['dataset']['name']
    num_classes = C.get()['num_classes']
    abspath = C.get()['abspath']
    sparsity = C.get()['sparsity']
    class_imbalance = C.get()['class_imbalance']
    n_aug = C.get()['n_aug']


    policy = search_policy(dataset=name, abspath=abspath)

    if path is None:
        train_dataset = load_dataset(path=name, data_dir=data_dir, data_files=data_files, split='train')
    elif path=='custom':
        train_dataset = load_dataset('csv', data_dir=data_dir, data_files=data_files, split='train')
    else: #glue path
        train_dataset = load_dataset(path=path, name=name, data_dir=data_dir, data_files=data_files, split='train')
        
    #Relabel classes
    try:
        train_dataset = refactor_classes(train_dataset, data_name, num_classes)
    except: #doesnt need refactored classes
        pass

    #Introducing sparsity and class imbalance
    train_dataset = rebalance_resize(train_dataset, class_imbalance, sparsity)
    
    augmented_train_dataset = augment(dataset=train_dataset, policy=policy, n_aug=n_aug, class_imbalance=class_imbalance, sparsity=sparsity)

    return augmented_train_dataset
