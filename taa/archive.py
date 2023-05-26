# Policy found on wiki_qa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .augmentation_method import augment_list, method_list, mode_list


def remove_deplicates(policies):
    """Remove duplicate policies. Considers duplicates if the 
    probabilities are _both_ either > or < 0.5"""
    seen = set()
    new_policies = []
    
    for policy in policies:
        elem = policy[0]  # Get the tuple from the list

        if elem[2] > 0.5: #if prob >0.5
            unique = tuple(elem[:2] + ('high',))
        else:
            unique = tuple(elem[:2] + ('low',))

        if unique in seen:
            continue

        new_policies.append(policy)
        seen.add(unique)
    
    return new_policies


def policy_decoder(augment, alpha, n_aug):
    '''
    Results['config'] == augment. This is the dictionary of results
    This function decodes integers back into function names
    :param augment: related to dict `space` and `config` in `exp_config`
    :return: list of policies 
    
    '''
    aug_list = augment_list(alpha, n_aug)
    policies = []
    for i in range(len(method_list())):
        for j in range(len(mode_list())):
            aug_method_idx = augment['aug_method_%d_%d' % (i, j)]
            aug_mode_idx = augment['aug_mode_%d_%d' % (i, j)]
            prob = augment['prob_%d_%d' % (i, j)]
            
            policies.append([(aug_list[j][0].__name__, aug_list[j][1].__name__, prob)])
            #remove elements of list to keep indexing
            if len(aug_list)>len(mode_list()):
                aug_list = aug_list[len(mode_list()):]
        
    return policies


def policy_decoder2(augment):
    '''
    Results['config'] == augment. This is the dictionary of results
    This function decodes integers back into function names
    :param augment: related to dict `space` and `config` in `exp_config`
    :return: single policy
    
    '''
    policies = []
    policies.append([(augment['aug_method'].__name__, augment['aug_mode'].__name__, augment['prob'])])
        
    return policies


#policy_map = {'imdb': imdb(), 'sst5': sst5(), 'trec': trec(), 'yelp2': yelp2(), 'yelp5': yelp5(), 'custom_data': imdb(), 'sst2': sst5()}
