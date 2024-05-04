import h3
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from custom_tokenizer import CustomDataset
import random

def convert_string_to_float_list(input_string):
    if input_string.strip('[]') == '':
        return []
    else:
        # Remove the square brackets and split the string by '],['
        cleaned_string = input_string.strip('[]')
        split_values = cleaned_string.split('],[')

        # Split each pair of values and convert them to float
        float_list = []
        for pair in split_values:
            values = pair.split(',')
            float_pair = [float(value) for value in values]
            float_list.append(float_pair)

        return float_list
    

def create_cluster_list(trajectory):
    cluster_list = []
    for point_list in trajectory:
        x = point_list[0]
        y = point_list[1]

        cluster = h3.geo_to_h3(x, y, 9)

        cluster_list.append(cluster)

    return cluster_list

def list_to_string(word_list):
    return ' '.join(word_list)

def createVocabulary(string_counts):
    vocab = {}

    vocab['[PAD]'] = 0
    vocab['[UNK]'] = 1
    vocab['[CLS]'] = 2
    vocab['[SEP]'] = 3
    vocab['[MASK]'] = 4

    for key in string_counts.keys():
        vocab[key] = len(vocab)

    return vocab

def saveVocab(vocab, data_path):
    # Specify the file path where you want to save the text file
    output_path = data_path + "vocab.txt"

    # Open the file in write mode and save the dictionary
    with open(output_path, "w") as file:
        json.dump(vocab, file)


def loadTrajectoryData(folder_path):
    # List to store individual DataFrames
    dfs = []

    # Iterate over each file in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            # Append the DataFrame to the list
            dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    trajectory_data = pd.concat(dfs, ignore_index=True)

    # Drop the index
    trajectory_data.reset_index(drop=True, inplace=True)

    return trajectory_data

def loadVocab(data_path):
    # Specify the file path of the JSON file
    json_file_path = data_path + "vocab.txt"

    # Open the JSON file and load its contents into a dictionary
    with open(json_file_path, "r") as json_file:
        vocab = json.load(json_file)

    return vocab

def createData(trajectory_data):
    train_df, test_df = train_test_split(trajectory_data, test_size=0.05, random_state=42)

    # Split the DataFrame into train and test sets
    eval_df, test_df = train_test_split(test_df, test_size=0.2, random_state=42)
    # Display the shapes of the train and test sets
    print("Train set shape:", train_df.shape)
    print("Eval set shape:", eval_df.shape)
    print("Test set shape:", test_df.shape)

    return train_df, eval_df, test_df

def createDatasets(train_df, eval_df, tokenizer, max_len):
    #Create the train and evaluation dataset
    train_dataset = CustomDataset(train_df['Trajectory_Sequence'], tokenizer, max_len)
    eval_dataset = CustomDataset(eval_df['Trajectory_Sequence'], tokenizer, max_len)

    return train_dataset, eval_dataset

def getEvaluationData(eval_file):
    df = pd.read_csv(eval_file)
    df.reset_index(drop=True, inplace=True)

    return df

def mask_random_word(text):
    words = text.split()
    random_index = random.randint(0, len(words) - 1)
    if random_index > 500:
        random_index = random.randint(0, 500)
    masked_word = words[random_index]
    words[random_index] = '[MASK]'
    words = words[:500]
    masked_text = ' '.join(words)
    return masked_text, masked_word

