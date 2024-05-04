import argparse
from tqdm import tqdm
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig, RobertaForMaskedLM, DataCollatorForLanguageModeling
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, pipeline
from utils import convert_string_to_float_list, create_cluster_list, getEvaluationData, list_to_string, \
createVocabulary, saveVocab, loadTrajectoryData, loadVocab, createData, createDatasets, mask_random_word
from custom_tokenizer import FixedVocabTokenizer
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.sql.functions import udf, size, explode, col, count
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

def createSparkSession():
    # Create a SparkSession
    spark = SparkSession.builder \
    .appName("My Spark") \
    .getOrCreate()

    return spark

def getModel(model_name):

    if model_name=='bert':
        config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=768, 
        num_hidden_layers=6, 
        num_attention_heads=12,
        max_position_embeddings=512
        )
        
        model = BertForMaskedLM(config)

        print('No of parameters: ', model.num_parameters())
    elif model_name=='roberta':
        config = RobertaConfig(
            vocab_size=len(vocab),
            hidden_size=768, 
            num_hidden_layers=6, 
            num_attention_heads=12,
            max_position_embeddings=512
        )
        
        model = RobertaForMaskedLM(config)
        print('No of parameters: ', model.num_parameters())
    else:
        print("Error: Incorrect Model Name")

    return model

def preprocessRawData(df):
    df_polyline = df.select('POLYLINE')

    convert_string_to_float_list_udf = udf(convert_string_to_float_list, ArrayType(ArrayType(FloatType())))

    df_trajectories = df_polyline.withColumn("Trajectory", convert_string_to_float_list_udf(df["POLYLINE"])).select("Trajectory")

    print(f'Total Number of Tajectories : {df_trajectories.count()}')

    create_cluster_list_udf = udf(create_cluster_list, ArrayType(StringType()))

    df_clusters = df_trajectories.withColumn("Trajectory_clusters", create_cluster_list_udf(df_trajectories["Trajectory"])).select("Trajectory_clusters")

    df_exploded = df_clusters.select(explode(col("Trajectory_clusters")).alias("string"))

    string_counts_df = df_exploded.groupBy("string").agg(count("*").alias("count"))

    string_counts = {row["string"]: row["count"] for row in string_counts_df.collect()}

    print(f'Total Number of Clusters : {len(string_counts)}')

    vocab = createVocabulary(string_counts)

    print(f'Total Number of Keys in Vocabulary : {len(vocab)}')

    #Filter empty trajectories
    df_clusters_new = df_clusters.filter(size("Trajectory_clusters") > 0)

    # Calculate average length
    average_length_df = df_clusters_new.selectExpr("avg(size(Trajectory_clusters)) as average_length")

    # Calculate maximum length
    max_length_df = df_clusters_new.selectExpr("max(size(Trajectory_clusters)) as max_length")

    # Calculate minimum length
    min_length_df = df_clusters_new.selectExpr("min(size(Trajectory_clusters)) as min_length")

    # Fetch results
    average_length = average_length_df.collect()[0]["average_length"]
    max_length = max_length_df.collect()[0]["max_length"]
    min_length = min_length_df.collect()[0]["min_length"]

    print(f'Average Trajectory Length : {average_length}')
    print(f'Maximum Trajectory Length : {max_length}')
    print(f'Minimum Trajectory Length : {min_length}')

    # Register the UDF
    list_to_string_udf = udf(list_to_string, StringType())

    # Apply the UDF to the DataFrame
    df_trajectory_seq = df_clusters_new.withColumn("Trajectory_Sequence", list_to_string_udf(df_clusters_new["Trajectory_clusters"])).select("Trajectory_Sequence")

    return df_trajectory_seq, vocab

def trainModel(model, model_folder, data_collator, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=model_folder,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #prediction_loss_only=True,
    )

    trainer.train()
    trainer.save_model(model_folder + 'saved/')

def evaluate(test_df):
    predicted_labels = []
    true_labels = []
    for index in tqdm(range(len(test_df))):
        # Access df['column_name'][index] or df.iloc[index, column_index] to access values
        #print(index)
        str = test_df['Trajectory_Sequence'][index]
        label = test_df['label'][index]
        true_labels.append(tokenizer(label, padding=True, truncation=True).input_ids[0])
        res = fill_mask(str)
        predicted_labels.append(res[0]['token'])

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average='weighted')

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print("Recall:", recall)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)

def saveTrajectories(df, data_path):
    output_path = data_path + "trajectory_seq.csv"

    # Save the DataFrame to a CSV file
    df.write.csv(output_path, header=True, mode="overwrite")

def createTokenizer(vocab, max_len):
    tokenizer = FixedVocabTokenizer(vocab, max_len=max_len)

    # tell your tokenizer about your special tokens
    tokenizer.add_special_tokens({
                'unk_token': '[UNK]',
                'pad_token': '[PAD]',
                'cls_token': '[CLS]',
                'sep_token': '[SEP]',
                'mask_token': '[MASK]'
    })

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-chat-hf', type=str, help='Model Name for training')
    parser.add_argument('--seed', default=0, type=int, help='pytorch seed')
    parser.add_argument('--res', default=9, type=int, help='Resolution for H3 Library')
    parser.add_argument('--dir_path', default='', type=str, help='Save Path for the model')
    parser.add_argument('--mode', type=str, help='Mode of Run')

    print(f'Cude is available : {torch.cuda.is_available()}')
    args = parser.parse_args()
    MODEL , SEED, RESOLUTION, DIR_PATH, MODE = (args.model , args.seed, args.res, args.dir_path, args.mode)
    DATA_PATH = DIR_PATH + 'Data/'
    MODEL_PATH = DIR_PATH + 'Models/'


    if(MODE=='preprocessing'):
        print('Ruuning the program in Preprocessing Mode.....')

        spark = createSparkSession()

        data = spark.read.csv(DATA_PATH+'train.csv', header=True, inferSchema=True)

        trajectory_seq,vocab = preprocessRawData(data)

        saveTrajectories(trajectory_seq,DATA_PATH)

        saveVocab(vocab, DATA_PATH)
    elif(MODE=='pretraining'):
        print('Ruuning the program in Pre-Training Mode.....')

        file_path = DATA_PATH + "trajectory_seq.csv"

        trajectory_data = loadTrajectoryData(file_path)

        vocab = loadVocab(DATA_PATH)

        model_max_len = 128

        tokenizer = createTokenizer(vocab, model_max_len)

        train_df, eval_df, test_df = createData(trajectory_data)

        #Saving the test dataset for evaluation
        test_df.to_csv(DATA_PATH + 'evaluation_data.csv', index=False)

        train_dataset, eval_dataset = createDatasets(train_df, eval_df, tokenizer, model_max_len)

        # Define the Data Collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        model = getModel(MODEL)

        trainModel(model, MODEL_PATH, data_collator, train_dataset, eval_dataset)
        

    elif(MODE=='evaluation'):
        print('Ruuning the program in Evaluation Mode.....')

        if MODEL=='bert':
            model = BertForMaskedLM.from_pretrained(MODEL_PATH + '/saved')
        elif MODEL=='roberta':
            model = RobertaForMaskedLM.from_pretrained(MODEL_PATH + '/saved')

        vocab = loadVocab(DATA_PATH)

        model_max_len = 128

        tokenizer = createTokenizer(vocab, model_max_len)

        fill_mask = pipeline(
            "fill-mask",
            model=model,
            tokenizer=tokenizer
        )

        eval_file_path = DATA_PATH + 'evaluation_data.csv'

        test_df = getEvaluationData(eval_file_path)

        # Apply the function to each row in the DataFrame
        test_df['Trajectory_Sequence'], test_df['label'] = zip(*test_df['Trajectory_Sequence'].map(mask_random_word))

        print(f'Total Trajectories in Test Set : {len(test_df)}')

        evaluate(test_df)
    else:
        print('Error: Please provide Correct Mode')
    



