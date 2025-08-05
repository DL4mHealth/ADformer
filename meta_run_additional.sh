#!/bin/bash

# Run scripts sequentially

# Patch Length Study
bash scripts/ADformer/supervised/ADformer/Ablation/Patch-Length-Study/S-1.sh

# Channel Number Study
bash scripts/ADformer/supervised/ADformer/Ablation/Channel-Number-Study/S-1.sh

# Overlap Study
bash scripts/ADformer/supervised/ADformer/Ablation/Overlap-Study/S-1.sh

# Length Study
bash scripts/ADformer/supervised/ADformer/Ablation/Length-Study/S-1.sh