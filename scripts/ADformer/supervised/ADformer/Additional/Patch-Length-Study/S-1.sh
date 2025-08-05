export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
# ADFTD
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Patch --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 2 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Patch --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 4 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Patch --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 8 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Patch --model ADformer --data MultiDatasets \
#--training_datasets ADFTD \
#--testing_datasets ADFTD \
#--patch_len_list 2,2,2 \
#--no_spatial_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Patch --model ADformer --data MultiDatasets \
#--training_datasets ADFTD \
#--testing_datasets ADFTD \
#--patch_len_list 4,4,4 \
#--no_spatial_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Patch --model ADformer --data MultiDatasets \
#--training_datasets ADFTD \
#--testing_datasets ADFTD \
#--patch_len_list 8,8,8 \
#--no_spatial_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Patch --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--patch_len_list 2,4,8 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15



# CAUEEG
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 2 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 4 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 8 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 2,2,2 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
#--training_datasets CAUEEG \
#--testing_datasets CAUEEG \
#--patch_len_list 2,2,2 \
#--no_spatial_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
#--training_datasets CAUEEG \
#--testing_datasets CAUEEG \
#--patch_len_list 4,4,4 \
#--no_spatial_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
#--training_datasets CAUEEG \
#--testing_datasets CAUEEG \
#--patch_len_list 8,8,8 \
#--no_spatial_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Patch --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--patch_len_list 2,4,8 \
--no_spatial_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15