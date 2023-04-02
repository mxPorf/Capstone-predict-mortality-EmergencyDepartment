export SM_CHANNEL_TRAINING=''
export SM_MODEL_DIR=''
export SM_OUTPUT_DATA_DIR=''
python train_and_deploy.py --learning_rate .01 --batch_size 64 --epochs 1 --data s3://sagemaker-us-east-1-927441871693/data/ --model_dir output_model/ --output_dir output/