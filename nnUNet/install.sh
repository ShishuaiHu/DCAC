#!/bin/bash
# @author: sshu
# @contact: sshu@mail.nwpu.edu.cn
# @file: install.sh
# @time: 2021/03/11
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
pip install hiddenlayer graphviz IPython
