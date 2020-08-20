conda env create -f environment.yml
conda activate rfs
pip uninstall pytorch-nlp
pip install git+https://github.com/ekinakyurek/PyTorch-NLP.git
