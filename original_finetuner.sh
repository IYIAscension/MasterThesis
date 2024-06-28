#!/bin/bash
#SBATCH -J mt5-LA-finetuning
#SBATCH -t 0:55:00
#SBATCH -p gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=32

module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
#Some packages may be missing here.
pip install --user --upgrade pip
pip install --user accelerate
pip install --user dynamo
pip install --user sentencepiece
pip install --user transformers
pip install --user deepspeed
pip install --user datasets
pip install --user torchmetrics

export MASTER_PORT=6000
export TMP_DIR=$(mktemp -d -p /scratch-shared)
cp -r $HOME/original/train_examples $TMP_DIR
cp -r $HOME/original/dev_examples $TMP_DIR

mkdir $TMP_DIR/checkpoints_0

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run --nnodes=8 --nproc_per_node=4 --master_addr=$HOSTNAME --master_port=$MASTER_PORT --node_rank=$SLURM_PROCID $HOME/original/model_finetuner.py --train_d $TMP_DIR/train_examples --dev_d $TMP_DIR/dev_examples --cp_d $TMP_DIR/checkpoints_0'

newest_cp=$(ls -t -d "$TMP_DIR/checkpoints_0"/*/ | head -n 1)
newest_cp=$(echo "$newest_cp" | sed 's:/*$::')
python $HOME/original/zero_to_fp32.py "$newest_cp" "$TMP_DIR/checkpoints_0/first_model.bin"
cp $TMP_DIR/checkpoints_0/first_model.bin $HOME/original/first_model.bin
