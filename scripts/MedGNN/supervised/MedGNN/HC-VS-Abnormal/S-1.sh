export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
# ADFTD
python -u run.py --method MedGNN --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-HC --model MedGNN --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method MedGNN --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CNBPM-HC --model MedGNN --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# P-ADIC
python -u run.py --method MedGNN --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-P-ADIC-HC --model MedGNN --data MultiDatasets \
--training_datasets P-ADIC \
--testing_datasets P-ADIC \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CAUEEG
python -u run.py --method MedGNN --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-HC --model MedGNN --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15