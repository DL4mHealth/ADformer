export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
# ADFTD
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-HC --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 4,8 --up_dim_list 38,76 \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CNBPM-HC --model ADformer --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--patch_len_list 4,8 --up_dim_list 38,76 \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# P-ADIC
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-P-ADIC-HC --model ADformer --data MultiDatasets \
--training_datasets P-ADIC \
--testing_datasets P-ADIC \
--patch_len_list 4,8 --up_dim_list 38,76 \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CAUEEG
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-HC --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 4,8 --up_dim_list 38,76 \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice hc_vs_abnormal --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15