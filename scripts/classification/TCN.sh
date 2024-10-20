export CUDA_VISIBLE_DEVICES=0,1,2,3

# ADSZ Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADSZ/ \
  --model_id ADSZ-Dep \
  --model TCN \
  --data AD2Dep \
  --e_layers 8 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADSZ/ \
  --model_id ADSZ-Indep \
  --model TCN \
  --data AD2Indep \
  --e_layers 8 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# APAVA Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/AFAVA-AD/ \
  --model_id AFAVA-AD-Dep \
  --model TCN \
  --data AD2Dep \
  --e_layers 8 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/AFAVA-AD/ \
  --model_id AFAVA-AD-Indep \
  --model TCN \
  --data AD2Indep \
  --e_layers 8 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# ADFD Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFD/ \
  --model_id ADFD-Dep \
  --model TCN \
  --data AD3Dep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFD/ \
  --model_id ADFD-Indep \
  --model TCN \
  --data AD3Indep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# CNBPM Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/CNBPM/ \
  --model_id CNBPM-Dep \
  --model TCN \
  --data AD3Dep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/CNBPM/ \
  --model_id CNBPM-Indep \
  --model TCN \
  --data AD3Indep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# Cognision-ERP Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Cognision-ERP/ \
  --model_id Cognision-ERP-Dep \
  --model TCN \
  --data COG2Dep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Cognision-ERP/ \
  --model_id Cognision-ERP-Indep \
  --model TCN \
  --data COG2Indep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# Cognision-rsEEG Dataset
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Cognision-rsEEG/ \
  --model_id Cognision-rsEEG-Dep \
  --model TCN \
  --data COG2Dep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Cognision-rsEEG/ \
  --model_id Cognision-rsEEG-Indep \
  --model TCN \
  --data COG2Indep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# ADFD and CNBPM Datasets, use 2 classes only for patient-independent setting
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/ADFD/ \
  --model_id ADFD-3To2Indep \
  --model TCN \
  --data AD3To2Indep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 1e-4 \
  --train_epochs 100 \
  --patience 10
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/CNBPM/ \
  --model_id CNBPM-3To2Indep \
  --model TCN \
  --data AD3To2Indep \
  --e_layers 8 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 1e-4 \
  --train_epochs 100 \
  --patience 10

