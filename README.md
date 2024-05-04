# TrajectIQ

TrajectIQ is a BERT based trajectory imputation system that addresses the challenge of sparse and incomplete trajectory datasets by inserting additional realistic points to enhance accuracy.

## Get The Code

To download the code run the command in terminal:

```python
git clone git@github.com:kushagraDixit/TrajectIQ.git
```

# Create Environment

Create conda environment with the given file

```python
conda env create --name envname --file=env_tiq.yml
```

## Creation Additional Directories

To run the code you need the training data and a directory structure that will store all the data and weights of Model. To create the directory stucture run the file create_directories.py. We need space of 7-8 GB for storing the model. Provide the path for directory with the argutents as given in the example:

```python
python create_directories.py {path_to_directory}/{folder_name}
```

## Running the Code

The code has 3 modes:
1. preprocessing - Prepares the data for pretraing of the model
2. pretraining - pretraining of the model
3. evaluation - evaluation of model on the test set

Given with the code is a file run.sh which can be run directly on the SLURM Scheduler with CHPC. You need to change the following configuration in the file:

```python
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/uufs/chpc.utah.edu/common/home/u1472614/miniconda3/lib
export MODEL='roberta' #Select between bert/roberta
export SEED=0
export RES=9 #Resolution of H3 clusters
export DIR_PATH='{path_to_directory (put '/' in the end)}'
export SCRDIR="{Path to a folder for logging}"
export WORKDIR="{Path to working directory}"
```

## Reading Results

Once the SLURM starts running the program it will create a file traject_iq-{Job_ID} through which you can follow the progress of the program. The run.sh runs the program in all 3 modes in a sequential manner. Once the program run is completed. to see the results to the directory used in the SLURM file for logging. There would be 3 Output files. 
 
1. Output1 - This file is for preprocessing mode and contains the some data statistics
2. Output2 - This file contain the outputs of model pre-training step.
3. Output3 - This file contain the evaluation results.
