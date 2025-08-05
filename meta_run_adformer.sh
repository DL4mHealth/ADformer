#!/bin/bash

# Run scripts sequentially

# ADformer
bash ./scripts/ADformer/supervised/ADformer/AD-VS-HC/S-1.sh
bash ./scripts/ADformer/supervised/ADformer/AD-VS-NonAD/S-1.sh
bash ./scripts/ADformer/supervised/ADformer/HC-VS-Abnormal/S-1.sh
bash ./scripts/ADformer/supervised/ADformer/Multi-Class/S-1.sh