export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
# ADFTD
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Overlap --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.0 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Overlap --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.2 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Overlap --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Overlap --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.8 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15



# CAUEEG
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Overlap --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.0 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Overlap --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.2 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Overlap --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.5 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Overlap --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 4 --up_dim_list 76 \
--segment_length 128 --overlapping 0.8 \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
