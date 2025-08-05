#!/bin/bash

# Run scripts sequentially

# BIOT
bash ./scripts/BIOT/supervised/BIOT/AD-VS-HC/S-1.sh
bash ./scripts/BIOT/supervised/BIOT/Multi-Class/S-1.sh

# Conformer
bash ./scripts/Conformer/supervised/Conformer/AD-VS-HC/S-1.sh
bash ./scripts/Conformer/supervised/Conformer/Multi-Class/S-1.sh

# EEGNet
bash ./scripts/EEGNet/supervised/EEGNet/AD-VS-HC/S-1.sh
bash ./scripts/EEGNet/supervised/EEGNet/Multi-Class/S-1.sh

# EEGInception
bash ./scripts/EEGInception/supervised/EEGInception/AD-VS-HC/S-1.sh
bash ./scripts/EEGInception/supervised/EEGInception/Multi-Class/S-1.sh

# Medformer
bash ./scripts/Medformer/supervised/ADformer/AD-VS-HC/S-1.sh
bash ./scripts/Medformer/supervised/ADformer/Multi-Class/S-1.sh

# MedGNN
bash ./scripts/MedGNN/supervised/MedGNN/AD-VS-HC/S-1.sh
bash ./scripts/MedGNN/supervised/MedGNN/Multi-Class/S-1.sh

# Transformer
bash ./scripts/Transformer/supervised/Transformer/AD-VS-HC/S-1.sh
bash ./scripts/Transformer/supervised/Transformer/Multi-Class/S-1.sh

# ManualFeature
bash ./scripts/ManualFeature/supervised/ManualFeature/AD-VS-HC/S-1.sh
bash ./scripts/ManualFeature/supervised/ManualFeature/Multi-Class/S-1.sh