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



logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def load_data(s3_bucket_name, prefix='data', file='cleaned_data.csv'):
    '''
    Loads pre-processed data from an S3 bucket
    '''
    key=os.path.join(prefix,file)
    
    client = boto3.client('s3')
    response = client.get_object(Bucket=s3_bucket_name, Key=key)
    body=response['Body']

    data = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    return df

def net():
    '''
    Retrieves a pretrained PyTorch Tabular model from s3
    '''
    model = retrieve_from_s3()
    return model
#     model = models.resnet50(pretrained=True)

#     for param in model.parameters():
#         param.requires_grad = False   

#     model.fc = nn.Sequential(
#                    nn.Linear(2048, 128),
#                    nn.ReLU(inplace=True),
#                    nn.Linear(128, 133))
#     return model


def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    
    #Obtain pre-processed data
    df = load_data(args.s3_bucket_name)
    
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
        learning_rate = args.learnig_rate
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
    print(accuracy_score(data_test['deceased'],pred_df['prediction']))
    
    logger.info(f"Testing Accuracy: {total_acc}")
    ##################################
    #Save the model
    logger.info("Saving Model")
    torch.save(tabular_model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))
    
    

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