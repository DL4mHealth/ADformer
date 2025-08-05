export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
# ADFTD
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Channel --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--up_dim_list 19 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Channel --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--up_dim_list 38 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Channel --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--up_dim_list 76 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Channel --model ADformer --data MultiDatasets \
#--training_datasets ADFTD \
#--testing_datasets ADFTD \
#--up_dim_list 19,19,19 \
#--no_temporal_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Channel --model ADformer --data MultiDatasets \
#--training_datasets ADFTD \
#--testing_datasets ADFTD \
#--up_dim_list 38,38,38 \
#--no_temporal_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Channel --model ADformer --data MultiDatasets \
#--training_datasets ADFTD \
#--testing_datasets ADFTD \
#--up_dim_list 76,76,76 \
#--no_temporal_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-ADFTD-Channel --model ADformer --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--up_dim_list 19,38,76 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15



# CAUEEG
python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Channel --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--up_dim_list 19 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Channel --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--up_dim_list 38 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Channel --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--up_dim_list 76 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Channel --model ADformer --data MultiDatasets \
#--training_datasets CAUEEG \
#--testing_datasets CAUEEG \
#--up_dim_list 19,19,19 \
#--no_temporal_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Channel --model ADformer --data MultiDatasets \
#--training_datasets CAUEEG \
#--testing_datasets CAUEEG \
#--up_dim_list 38,38,38 \
#--no_temporal_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15
#
#python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Channel --model ADformer --data MultiDatasets \
#--training_datasets CAUEEG \
#--testing_datasets CAUEEG \
#--up_dim_list 76,76,76 \
#--no_temporal_block \
#--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
#--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

python -u run.py --method ADformer --task_name supervised --is_training 1 --root_path ./dataset/128Hz/ --model_id S-CAUEEG-Channel --model ADformer --data MultiDatasets \
--training_datasets CAUEEG \
--testing_datasets CAUEEG \
--up_dim_list 19,38,76 \
--no_temporal_block \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 --classify_choice ad_vs_hc --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15