import pandas as pd
from io import StringIO
import boto3
import argparse
import os
import logging
import sys

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


####################
S3_BUCKET_NAME='sagemaker-us-east-1-927441871693'
####################

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def model_fn(model_dir):
    '''
    Retrieves a pretrained PyTorch Tabular model from s3
    '''
    # import torch
    # path=os.path.join(model_dir, "model.pth")
    # key=os.path.join('model','model.pth')
    # model_object = retrieve_s3_object(key)
    # with open('model.pth', 'w') as f:
    #     f.write(model_object.read().decode('utf-8'))
    # model = torch.load('model.pth')
    return TabularModel.load_model(model_dir)
    # return model

def upload_to_s3(file_name, s3_prefix):
    client = boto3.client('s3')
    object_name=os.path.join(s3_prefix, file_name)
    try:
        response = client.upload_file(file_name, S3_BUCKET_NAME, object_name)
    except ClientError as e:
        print(e)
        
def retrieve_s3_object(key):
    client = boto3.client('s3')
    response = client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    return response['Body']

def load_data(prefix='data', file='cleaned_data.csv'):
    '''
    Loads pre-processed data from an S3 bucket
    '''
    key=os.path.join(prefix,file)
    body = retrieve_s3_object(key)
    data = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    return df




def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    
    #Obtain pre-processed data
    df = load_data()
    
    #Divide (randomly) into train and test datasets
    data_train, data_test = train_test_split(df, test_size=.2, stratify=df['deceased'])
    
    #Configure Tabular Model
    features = df.drop(columns='deceased')
    continuous_columns=list(features.columns)
    data_config = DataConfig(
        target=['deceased'],
        continuous_cols=continuous_columns,
        categorical_cols=[]
    )
    trainer_config=TrainerConfig(
        auto_lr_find=True, 
        batch_size=args.batch_size,
        max_epochs=args.epochs,
    )
    optimizer_config=OptimizerConfig()
    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="1024-512-512",  
        activation="LeakyReLU", 
        learning_rate = args.learning_rate
    )
    #Create the model with the previous configuration
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    #Train the model
    logger.info("Starting Model Training")
    tabular_model.fit(train=data_train, validation=data_test)
    #Obtain accuracy metric
    logger.info("Testing Model")
    val_cases = data_test.drop(columns='deceased')
    pred_df = tabular_model.predict(val_cases)
    ################################
    total_acc = accuracy_score(data_test['deceased'],pred_df['prediction'])
    logger.info(f"Testing Accuracy: {total_acc}")
    ##################################
    #Save the model
    logger.info("Saving Model")
    # save_path = os.path.join(args.model_dir, "model.pth")
    # tabular_model.save_model(args.model_dir, kind='pytorch')
    tabular_model.save_model(args.model_dir)
    
    # upload_to_s3("model.pth", s3_prefix='model')
    
    #############################
    ############################
    #  FOR TESTS
    # retreived_model = model_fn('output')
    # print(retreived_model.evaluate())
    
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    print(args)
    
    main(args)