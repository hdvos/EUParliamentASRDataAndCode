#!/bin/bash

# reset everything
#SBATCH --export=NONE

# number of nodes
#SBATCH --nodes=1# 

# number of tasks per node
#SBATCH --ntasks=1

# time you think your job will take (process will end after this time!)
#SBATCH --time=24:00:00

# number of gpus you want to run your job on
#SBATCH --gres=gpu:4

# memory needed
#SBATCH --mem=240 

# which partition you want to send the job to (view partitions with 'sinfo')
# # SBATCH --partition=testing
#SBATCH --partition=gpu-medium

# email adress to contact with notifications
#SBATCH --mail-user=h.p.de.vos@fgga.leidenuniv.nl

# when to send notifications <BEGIN|END|FAIL|REQUEUE|ALL>
#SBATCH --mail-type="FAIL"
#SBATCH --mail-type="END"

# Load modules

#Pytorch

#Python 3.7+

echo "DATE"
date

echo "============LOAD TensorFlow============"
#module load Miniconda3/4.7.10
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
echo "============LOAD PyTorch============"
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4

export PYTHONPATH=$PATH:/data/voshpde/python_modules
echo "PYTHONPATH:"
echo $PYTHONPATH


# Install
pip install pyannote.core --target=/data/voshpde/python_modules
pip install pyannote.audio==1.1.1 --target=/data/voshpde/python_modules
pip install pydub --target=/data/voshpde/python_modules
pip install librosa --target=/data/voshpde/python_modules
pip install transformers --target=/data/voshpde/python_modules --upgrade
pip install pyctcdecode --target=/data/voshpde/python_modules
pip install https://github.com/kpu/kenlm/archive/master.zip --target=/data/voshpde/python_modules
pip install jiwer --target=/data/voshpde/python_modules
pip install pywer --target=/data/voshpde/python_modules
pip install torchaudio==0.6.0 --target=/data/voshpde/python_modules


# cd to wd
cd /data/voshpde/pipeLine/hotwords_experiment


# run
python hotwords_experiment_0_2.py &
python hotwords_experiment_3_5.py &
python hotwords_experiment_6_8.py & 
python hotwords_experiment_8_10.py &
wait