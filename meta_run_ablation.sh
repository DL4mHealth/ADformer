#!/bin/bash

# Run scripts sequentially

# No Inter Attention
bash scripts/ADformer/supervised/ADformer/Ablation/No-Inter-Attention/S-1.sh

# No Temporal Block
bash scripts/ADformer/supervised/ADformer/Ablation/No-Temporal/S-1.sh

# No Spatial Block
bash scripts/ADformer/supervised/ADformer/Ablation/No-Spatial/S-1.sh