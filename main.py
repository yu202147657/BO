import sys
import json
from theconf import Config as C, ConfigArgumentParser
import os
import copy
import csv
import torch
import shutil
from pystopwatch2 import PyStopwatch
from taa.common import get_logger, add_filehandler
from taa.search import get_path, search_policy, train_model_parallel
import logging

w = PyStopwatch()
logging.basicConfig(level=logging.INFO)
logger = get_logger('Text AutoAugment')

parser = ConfigArgumentParser(conflict_handler='resolve')
parser.add_argument('--until', type=int, default=5)
parser.add_argument('--abspath', type=str, default='/home/jovyan/examples/examples/pytorch/MPhil_BO')
parser.add_argument('--num-search', type=int, default=1)
parser.add_argument('--n-aug', type=int, default=8,
                    help='n-aug for each given sample')
parser.add_argument('--trail', type=int, default=1, help='trail')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--sparsity', type=int, help='dataset size')
parser.add_argument('--class_imbalance', type=float, help='Percent of dataset that is majority class')
parser.add_argument('--alpha', type=float, default=0.05, help='strength of augmentation')

args = parser.parse_args()

data_name = C.get()['dataset']['name']
model_type = C.get()['model']['type']
abspath = args.abspath
add_filehandler(logger, os.path.join('logs', '%s_%s_seed%d_n-aug%d_alpha%.2f_sparsity%d_class_imbalance%.2f_it%d.log' %
                                     (data_name, model_type, args.seed, args.n_aug, args.alpha, args.sparsity, args.class_imbalance, args.num_search)))
logger.info('Configuration:')
logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))

copied_c = copy.deepcopy(C.get().conf)

# Train without Augmentations
logger.info('-' * 59)
logger.info('----- Train without Augmentations seed=%3d -----' % (args.seed))
logger.info('-' * 59)
torch.cuda.empty_cache()
w.start(tag='train_no_aug')
model_path = get_path(data_name, model_type, 'pretrained_trail%d_seed%d_sparsity%d' %
                       (args.trail, args.seed, args.sparsity))
logger.info('Model path: {}'.format(model_path))

pretrain_results = train_model_parallel('pretrained_trail%d_n-aug%d' %
                                        (args.trail, args.n_aug), copy.deepcopy(copied_c),
                                        augment=None, save_path=model_path,
                                        only_eval=True if os.path.isfile(model_path) else False, save_loss=True)
logger.info('Getting results:')
for train_mode in ['pretrained']:
    avg = 0.
    r_model, r_dict = pretrain_results
    log = ' '.join(['%s=%.4f;' % (key, r_dict[key]) for key in r_dict.keys()])
    logger.info('[%s] ' % train_mode + log)
w.pause('train_no_aug')
logger.info('Processed in %.4f secs' % w.get_elapsed('train_no_aug'))
shutil.rmtree(model_path)
if args.until == 1:
    sys.exit(0)

# Search Test-Time Augmentation Policies
logger.info('-' * 51)
logger.info('----- Search Test-Time Augmentation Policies -----')
logger.info('-' * 51)

w.start(tag='search')
final_policy = search_policy(dataset=data_name, abspath=abspath, num_search=args.num_search)
w.pause(tag='search')

# Train with Augmentations
logger.info('-' * 94)
logger.info('----- Train with Augmentations model=%s dataset=%s aug=%s -----' %
            (model_type, data_name, C.get()['aug']))
logger.info('-' * 94)
torch.cuda.empty_cache()
w.start(tag='train_aug')

augment_path = get_path(data_name, model_type, 'augment_trail%d_n-aug%d_seed%d' %
                         (args.trail, args.n_aug, args.seed))
logger.info('Getting results:')
final_results = train_model_parallel('augment_trail%d_n-aug%d' %
                                     (args.trail, args.n_aug), copy.deepcopy(copied_c),
                                     augment=final_policy, save_path=augment_path, only_eval=False)

for train_mode in ['augment']:
    avg = 0.
    r_model, r_dict = final_results
    log = ' '.join(['%s=%.4f;' % (key, r_dict[key]) for key in r_dict.keys()])
    logger.info('[%s] ' % train_mode + log)
w.pause('train_aug')
logger.info('Processed in %.4f secs' % w.get_elapsed('train_aug'))

logger.info(w)
shutil.rmtree(augment_path)

# Open a CSV file in write mode
with open(os.path.join('results', '%s_%s_seed%d_n-aug%d_alpha%.2f_sparsity%d_class_imbalance%.2f_policy%s_it%d.csv' %
                                     (data_name, model_type, args.seed, args.n_aug, args.alpha, args.sparsity, args.class_imbalance, final_policy, args.num_search)), 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=pretrain_results[1].keys())
    # Write header
    writer.writeheader()
    # Write data
    writer.writerow(pretrain_results[1])
    writer.writerow(final_results[1])
    