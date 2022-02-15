conda create --name subreg python=3.8
conda activate subreg
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install tensorboard scipy pandas

# Below versions work as well.
# conda create --name subspace python=3.6
# conda activate subspace
# conda install pytorch==1.7.0 torchvision==0.8.0 tensorboard=1.12.2 cudatoolkit=9.0 -c pytorch
# conda install scipy=1.5.0 pandas=1.1.2