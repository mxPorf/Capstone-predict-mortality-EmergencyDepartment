import pandas as pd
from io import StringIO
import boto3
import argparse
import os
import logging
import sys

import lightgbm as lgb
from numpy import argmax
 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from joblib import dump, load


####################
S3_BUCKET_NAME='sagemaker-us-east-1-927441871693'
####################

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def model_fn(model_dir):
    logger.info('In model fn')
    file = os.path.join(model_dir, 'model.joblib')
    model = load(file)
    return model
    
def predict_fn(input_object, model):
    logger.info('In predict fn')
    prediction = model.predict(input_object)
    prediction = argmax(prediction, axis=1)
    return prediction
        
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
    
    #Obtain pre-processed data
    df = load_data()
    
    features = df.drop(columns='deceased')
    target = df['deceased']
    
    #Divide (randomly) into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=.2, stratify=target)
    
    #Define model parameters
    params = {
        'objective': 'multiclass',
        "num_class": 2,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "early_stopping_rounds": args.early_stopping_rounds,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
    }
    
    #Load data into format required by lgb framework
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
 
    #Train the model
    model = lgb.train(params,
                      train_set=lgb_train,
                      valid_sets=lgb_eval)
    #predict
    y_pred = model.predict(x_test)
    y_pred = argmax(y_pred, axis=1)
    total_acc = accuracy_score(y_test, y_pred)
    logger.info(f"Testing Accuracy: {total_acc}")
    #Save the model
    logger.info("Saving Model")
    save_path = os.path.join(args.model_dir, "model.joblib")
    dump(model, save_path)
    
    if args.test:
        model_test=model_fn(args.model_dir)
        predictions = predict_fn(x_test, model_test)
        print(f'acc: {accuracy_score(y_test, y_pred)}')
        
    
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()        
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--early_stopping_rounds', type=int)
    parser.add_argument('--num_leaves', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    print(args)
    
    main(args)