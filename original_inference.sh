#!/bin/bash
#SBATCH -J mt5-LA-inference
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus=2

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
pip install --user --upgrade pip
pip install --user nltk
pip install --user accelerate
pip install --user dynamo
pip install --user sentencepiece
pip install --user transformers
pip install --user triton
pip install --user deepspeed

rm $HOME/.triton/autotune/*.lock

cp -r $HOME/original/test_texts "$TMPDIR"
cp -r $HOME/original/first_model.bin "$TMPDIR"

mkdir "$TMPDIR"/riddlecoref_coref_texts

python -m torch.distributed.run --nproc_per_node=2 $HOME/original/model_tester.py "$TMPDIR"/test_texts "$TMPDIR"/riddlecoref_coref_texts "$TMPDIR"/first_model.bin

cp -r "$TMPDIR"/riddlecoref_coref_texts $HOME/original
