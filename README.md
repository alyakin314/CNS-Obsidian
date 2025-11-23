# CNS-Obsidian
## Training Multimodal Neurosurgical Assistant

## Installation

To install **CNS-Obsidian** with all its dependencies (including PyTorch + CUDA
 12.1 wheels) in a Python 3.12 environment, follow these steps:

1. Create and activate a Python 3.12 environment:
```bash
conda create -n cns_obsidian python=3.12 -y
conda activate cns_obsidian
```
2. Clone the repository and install with the extra index url for CUDA 12.1 wheels.
```bash
git clone git@github.com:nyuolab/CNS-Obsidian.git
cd CNS-Obsidian
pip install . --extra-index-url https://download.pytorch.org/whl/cu121 --editable
```
We use --extra-index-url so that PyTorch and its associated CUDA 12.1 wheels can be
downloaded from the official PyTorch channel, while all other packages come from PyPI.
