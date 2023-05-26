import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import ast

def computation_graph(folder_path):
    
    
    dict_f1 = {'amazon': [], 'wiki': []}
    dict_time = {'amazon': [], 'wiki': []}
    dict_opt = {'amazon': [], 'wiki': []}
    iterations_lst = [15, 30, 50, 100]
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            file_path = os.path.join(folder_path, filename)  # Get the full file path

            #get the iteration:
            result = re.search('_it(.*).csv', filename)
            it = int((result.group(1)))

            df = pd.read_csv(file_path, index_col=False)
            min_loss = df['eval_loss'].idxmin()
            best_macro_f1 = df['eval_macro_f1'].max()
            opt = df['opt_object'].max()
            total_time = df['elapsed_time'].sum()
            
            if 'amazon' in filename:
                dict_f1['amazon'].append((it, best_macro_f1))
                dict_time['amazon'].append((it, total_time))
                dict_opt['amazon'].append((it, opt))
            else:
                dict_f1['wiki'].append((it, best_macro_f1))
                dict_time['wiki'].append((it, total_time))
                dict_opt['wiki'].append((it, opt))
            
    #Sort by iteration in dict:
    for dictionary in [dict_f1, dict_time, dict_opt]:
        for key, value in dictionary.items():
            # Sort the list by the first element in each tuple
            sorted_list = sorted(value, key=lambda x: x[0])

            # Extract the second element from each tuple
            result_list = [item[1] for item in sorted_list]

            # Update the value in the dictionary
            dictionary[key] = result_list
    
    print(dict_f1, dict_time, dict_opt)

    fig, ax = plt.subplots()
    for dataset in ['amazon', 'wiki']:
        ax.plot(iterations_lst, dict_f1[dataset], marker='o', label = f'{dataset}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Evaluation Macro f1')
    #ax.set_ylabel('Macro f1')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim(0, 100)
    plt.savefig(f'plots/line_plots/computation_graph.png', bbox_inches="tight")
    
    fig, ax = plt.subplots()
    for dataset in ['amazon', 'wiki']:
        ax.plot(iterations_lst, dict_time[dataset], marker='o', label = f'{dataset}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time')
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig(f'plots/line_plots/computation_graph_time.png', bbox_inches="tight")
    plt.show()

                    
            
#def bo_efficiency(folder_path, sparsity, dataset):
def bo_efficiency(folder_path, dataset):
    #sort by time stampp, plot eval loss for each file in folder
    
    fig, ax = plt.subplots()
    
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        
        #if filename.endswith('.csv'):
        #if filename.endswith('.csv') and sparsity == int(re.search(r'sparsity([^_]+)', filename).group(1)) and dataset == re.search(r'^([^_]+)', filename).group(1):  # Check if the file is a CSV file
        if filename.endswith('.csv') and dataset == re.search(r'^([^_]+)', filename).group(1):  # Check if the file is a CSV file
            file_path = os.path.join(folder_path, filename)  # Get the full file path
            
            df = pd.read_csv(file_path, index_col=False)
            #sort by order of time
            df = df.sort_values('timestamp')
            iterations = list(range(0, len(df)))
                              
            #ax.plot(iterations, df['eval_loss'].values, label=f'{filename}')
            #ax.plot(iterations, df['opt_object'].values, label=f'{filename}')
            ax.plot(iterations, df['eval_macro_f1'].values, label=f'{filename}')
            
    
    ax.set_xlabel('Iterations')
    #ax.set_ylabel('Evaluation Loss')
    #ax.set_ylabel('Evaluation Accuracy')
    ax.set_ylabel('Evaluation Macro f1')
    #plt.grid()
    #for vline in [15, 30, 50, 100, 200]:
    #    plt.axvline(x=vline, color='gray')
    #    plt.text(vline, 100, f'{vline}', color='gray', ha='center', va='bottom')

    plt.ylim(0, 100)
    #plt.ylim(0, 1.25)
    #plt.savefig(f'plots/line_plots/bo_efficiency_{sparsity}_{dataset}.png', bbox_inches="tight")
    plt.savefig(f'plots/line_plots/bo_efficiency_{dataset}.png', bbox_inches="tight")
    plt.show()
    
    
def extract_top_n(folder_path, n):
    """Extract the top n policies from BO and their eval loss + macro f1"""
    
    first_iteration = True
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            file_path = os.path.join(folder_path, filename)  # Get the full file path
            
            df = pd.read_csv(file_path, index_col=False)
            #sort by order of time
            #df = df.sort_values('opt_object',  ascending=False)
            df = df.sort_values('eval_macro_f1',  ascending=False)
            df = df.iloc[0:n]
            
            # extract the policy
            pattern = r"'aug': \[\[(.*?)\]\]"
            # Apply the extraction using the pattern to the entire column
            df['policy'] = df['config'].str.extract(pattern, expand=False)
           
            columns_to_keep = ['eval_loss', 'eval_macro_f1', 'opt_object', 'policy']
            df = df.loc[:, columns_to_keep].reset_index(drop=True)  
            
            df_reshaped = df.stack().to_frame().transpose()

            # Rename the columns to include the appended index
            index_level_0 = df_reshaped.columns.get_level_values(0)
            index_level_1 = df_reshaped.columns.get_level_values(1)

            new_columns = [f"{col}_{idx}" for col, idx in zip(index_level_1, index_level_0)]

            df_reshaped.columns = new_columns
            df_reshaped['sparsity'] = re.search(r'sparsity([^_]+)', filename).group(1)
            df_reshaped['class_imbalance'] = re.search(r'_class_imbalance([\d.]+)(?!\.)', filename).group(1)
            df_reshaped['dataset'] = re.search(r'^([^_]+)', filename).group(1) 

            if first_iteration: #write header
                df_reshaped.to_csv(f'top{n}.csv', mode='a', index=False)
                first_iteration = False
            else:
                df_reshaped.to_csv(f'top{n}_f1.csv', mode='a', index=False, header=False)

def result_closeness(folder_path, dataset, sparsity):
    fig, ax = plt.subplots()
    
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        
        #if filename.endswith('.csv'):
        if filename.endswith('.csv')  and sparsity == int(re.search(r'sparsity([^_]+)', filename).group(1)) and dataset == re.search(r'^([^_]+)', filename).group(1):  
            file_path = os.path.join(folder_path, filename)  # Get the full file path
            
            df = pd.read_csv(file_path, index_col=False)
            #df = df.sort_values('opt_object',  ascending=False)
            df = df.sort_values('eval_macro_f1',  ascending=False)
            
            # extract the policy
            pattern = r"'aug': \[\[(.*?)\]\]"
            # Apply the extraction using the pattern to the entire column
            df['policy'] = df['config'].str.extract(pattern, expand=False)
            
            # Convert strings to tuples, check the condition, and update values in the DataFrame
            for index, row in df.iterrows():
                tuple_value = ast.literal_eval(row['policy'])
                if tuple_value[2] < 0.5:
                    df.at[index, 'policy'] = 'None'
                else:
                    df.at[index, 'policy'] = tuple(tuple_value[:2])
            df = df.drop_duplicates(subset=['policy'], keep='first')
            columns_to_keep = ['eval_loss', 'eval_macro_f1', 'opt_object', 'policy']
            df = df.loc[:, columns_to_keep].reset_index(drop=True)
            
            #cap df at 8 rows bc that's max unique policy in the df
            df = df.iloc[0:8]
            
            #add in class imbalance
            imbalance = re.search(r'_class_imbalance([\d.]+)(?!\.)', filename).group(1)
            df.to_csv(f'results/result_closeness/{dataset}_{sparsity}_{imbalance}.csv')
            
            ax.plot(list(range(1, 9)), df['eval_macro_f1'].values, marker='o', label=f'{imbalance}')
    
    ax.set_xlabel('Rank')
    ax.set_ylabel('Evaluation Macro f1')
    plt.grid()
    plt.ylim(30, 100)
    plt.legend()
    plt.savefig(f'plots/line_plots/result_closeness_{dataset}_{sparsity}.png', bbox_inches="tight")
    plt.show()

def forget_vs_aug(folder_path, dataset, sparsity):
    fig, ax = plt.subplots()
    forget = []
    aug = []
    imbalance_lst = [0.5, 0.75, 0.85]
    
    # sort file path by class imbalance
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.csv')]
    class_imbalances = [filename.split('_class_imbalance')[1].split('.csv')[0] for filename in os.listdir(folder_path) if filename.endswith('.csv')]
    sorted_file_paths = [file_path for _, file_path in sorted(zip(class_imbalances, file_paths)) if file_path.endswith('.csv')]
    
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        
        #if filename.endswith('.csv'):
        if filename.endswith('.csv')  and sparsity == int(re.search(r'sparsity([^_]+)', filename).group(1)) and dataset == re.search(r'^([^_]+)', filename).group(1):  
            file_path = os.path.join(folder_path, filename)  # Get the full file path
            
            df = pd.read_csv(file_path, index_col=False)
            df = df.sort_values('eval_macro_f1',  ascending=False)
            
            # extract the policy
            pattern = r"'aug': \[\[(.*?)\]\]"
            # Apply the extraction using the pattern to the entire column
            df['policy'] = df['config'].str.extract(pattern, expand=False)
            
            # Convert strings to tuples, check the condition, and update values in the DataFrame
            for index, row in df.iterrows():
                tuple_value = ast.literal_eval(row['policy'])
                #only keep first element in policy tuple -> aug_method
                df.at[index, 'policy'] = tuple_value[0]
            forget_idx = df.index[df['policy'] == "influential_majority_forget"].min()
            aug_idx = df.index[df['policy'] == "influential_minority_aug"].min()
            
            forget.append(df.loc[forget_idx, 'eval_macro_f1'])
            aug.append(df.loc[aug_idx, 'eval_macro_f1'])
    
    ax.plot(imbalance_lst, forget, marker='o', label=f'Forget Influential Majority')
    ax.plot(imbalance_lst, aug, marker='o', label=f'Augment Influential Minority')
    
    ax.set_xlabel('Class Imbalance')
    ax.set_ylabel('Evaluation Macro f1')
    plt.grid()
    plt.ylim(30, 100)
    plt.legend()
    plt.savefig(f'plots/line_plots/aug_v_forget_{dataset}_{sparsity}.png', bbox_inches="tight")
    plt.show()
            
            
if __name__ == '__main__':
    #bo_efficiency('ray_results', 2000, 'wiki')
    computation_graph('results/computation_results')
    #bo_efficiency('results/computation_results', 'wiki')
    bo_efficiency('results/computation_results', 'amazon')
    #extract_top_n('ray_results', 5)
    #for dataset in ['amazon', 'wiki']:
    #    for sparsity in [500, 2000, 5000]:
    #        result_closeness('ray_results', dataset, sparsity)
    #forget_vs_aug('ray_results', 'amazon', 500)