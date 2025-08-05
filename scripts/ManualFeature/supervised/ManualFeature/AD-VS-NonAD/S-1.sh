export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
# ADFTD
python -u run.py --method ManualFeature --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-AD --model ManualFeature --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--batch_size 512 --classify_choice ad_vs_nonad --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method ManualFeature --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CNBPM-AD --model ManualFeature --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--batch_size 512 --classify_choice ad_vs_nonad --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# P-ADIC
python -u run.py --method ManualFeature --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-P-ADIC-AD --model ManualFeature --data MultiDatasets \
--training_datasets P-ADIC \
--testing_datasets P-ADIC \
--batch_size 512 --classify_choice ad_vs_nonad --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CAUEEG
python -u run.py --method ManualFeature --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-AD --model ManualFeature --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--batch_size 512 --classify_choice ad_vs_nonad --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15