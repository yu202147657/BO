import pandas as pd

def split(data_name, n = 1000):
    """Split data into train and test"""
    """Take 1000 rows for test set"""
    if data_name == "wiki":
        file_name = "taa/Data/clean_sentences_small_wiki"
    elif data_name == "amazon":
        file_name = "taa/Data/small_amazon"
    
    df = pd.read_csv(f"{file_name}.csv")
    test_df = df.sample(n=n)
    train_df = df.drop(test_df.index)
    
    test_df.to_csv(f"{file_name}_test.csv", index=False)
    train_df.to_csv(f"{file_name}_train.csv", index=False)
    

    

if __name__ == '__main__':
    split("wiki")
    split("amazon")