#!/usr/bin/env bash
#SBATCH --job-name='gbobrovskih.train_unimvsnet'
#SBATCH --output=%x@%A_%a.out
#SBATCH --error=%x@%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

set -ex

SINGULARITY_SHELL=/bin/bash \
singularity shell --nv \
    --bind /gpfs/gpfs0/3ddl/projects/sk3d/g.bobrovskih/logs/unimvsnet/:/home/logs \
    --bind /gpfs/gpfs0/3ddl/projects/sk3d/g.bobrovskih/dtu_training:/mnt/datasets/dtu \
    --bind /gpfs/gpfs0/3ddl/datasets/BlendedMVG/:/mnt/datasets/BlendedMVG \
    --bind /gpfs/gpfs0/3ddl/projects/sk3d/g.bobrovskih/sk3d.experiments:/home/src \
    /gpfs/gpfs0/3ddl/projects/sk3d/g.bobrovskih/gbobrovskikh.unimvsnet_cuda11.3-2022-10-11-34da1f614da5.sif << EOF

cd /home/src/UniMVSNet/src
./scripts/dtu_train.sh
    
EOF
