export SM_CHANNEL_TRAINING=''
export SM_MODEL_DIR=''
export SM_OUTPUT_DATA_DIR=''
python train_and_deploy.py --learning_rate .01 --num_iterations 1 --early_stopping_rounds 2 --num_leaves 3 --max_depth 10 --data s3://sagemaker-us-east-1-927441871693/data/ --model_dir output_model/ --output_dir output/ --test 1