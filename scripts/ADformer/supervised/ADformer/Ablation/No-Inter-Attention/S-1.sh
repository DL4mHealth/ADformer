export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
# ADFTD
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-ADHC-No-Inter --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 2 --up_dim_list 19 \
--no_inter_attn \
--segment_length 128 --overlapping 0.5 \
--e_layers 6 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CNBPM
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CNBPM-ADHC-No-Inter --model ADformer --data MultiDatasets \
--training_datasets CNBPM \
--testing_datasets CNBPM \
--patch_len_list 4,8 --up_dim_list 38,76 \
--no_inter_attn \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# P-ADIC
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-P-ADIC-ADHC-No-Inter --model ADformer --data MultiDatasets \
--training_datasets P-ADIC \
--testing_datasets P-ADIC \
--patch_len_list 4,8 --up_dim_list 38,76 \
--no_inter_attn \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# CAUEEG
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-ADHC-No-Inter --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 4,8,16 --up_dim_list 38,76,152 \
--augmentations none,mask0.2 \
--no_inter_attn \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15