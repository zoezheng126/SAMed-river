# Quick Start

git clone this repo `git clone https://github.com/zoezheng126/SAMed-river.git`\
checkout to development branch `git checkout development`\
Request SAMed_input dataset from Zoe (zoezheng126@hotmail.com) and move it to current directory \
Download sam_vit_b_01ec64.pth and LoRA8.pth to checkpoints folder \
Run `bash rivertest.sh` to get test results in `outputs/LoRA_0926`. Estimated time 5 mins using one Delta A40 node \

# Training
start training `bash rivertrain.sh` \
In another terminal, run `tensorboard --log-dir <model_output/your-train-session-name/log> --host your-delta-ip-address` (this may not work) \
After training, you can copy the new checkout point to checkpoints folder and update the bash command in rivertest.sh 





To Chenhui:
Dataset (delta path): /projects/bbkc/zoezheng126/SAMed-river/SAMed_input

1. Connect to GPU
2. conda activate SAMed (conda init with environment.yml)
