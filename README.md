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


```bash
.
├── README.md
├── requirements.txt
├── setup.py
├── model_printout.txt
├── cns_obsidian
│   ├── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── base_journal_dataset.py
│   │   ├── base_multimodal_dataset.py
│   │   ├── cns_dataset.py
│   │   ├── llava_med_dataset.py
│   │   └── pmc_oa_dataset.py
│   ├── instruct
│   │   ├── __init__.py
│   │   ├── api_call_processor.py
│   │   ├── api_calls_maker_ddx.py
│   │   ├── api_calls_maker_ift.py
│   │   ├── api_calls_maker_mc.py
│   │   ├── prompt_generator.py
│   │   ├── to_ask_a_question.py
│   │   ├── to_give_a_diagnosis.py
│   │   ├── to_make_a_choice.py
│   │   └── to_filter_a_dataset
│   │       ├── __init__.py
│   │       ├── image_labeling_app.py
│   │       ├── manual_filtering.ipynb
│   │       ├── pca_sorter.py
│   │       └── resnet_features.py
│   ├── slurm
│   │   ├── run_stage_1_med_align.slurm
│   │   ├── run_stage_2_med_finetune.slurm
│   │   └── run_stage_3_cns_finetune.slurm
│   ├── train
│   │   ├── obsidian_stage_1_med_align.py
│   │   ├── obsidian_stage_2_med_finetune.py
│   │   └── obsidian_stage_3_cns_finetune.py
│   └── utils
│       ├── __init__.py
│       ├── distributed.py
│       ├── io.py
│       └── utils.py
└── notebooks
    ├── examples
    │   ├── data_processing_example_radiopaedia.ipynb
    │   ├── example_dataset.json
    │   ├── fa4d9ef7ea69526338e3cff15d8434_big_gallery.jpeg
    │   ├── running-cns-obsidian.ipynb
    │   └── running-llava-next-med-olab.ipynb
    ├── figure_making
    │   ├── mcqs_figure_3.ipynb
    │   ├── obsidian_figure_2.ipynb
    │   ├── obsidian_figure_3.ipynb
    │   ├── obsidian_figure_5_anonymized.ipynb
    │   ├── obsidian_figure_s1.ipynb
    │   ├── obsidian_figure_s345.ipynb
    │   └── figures
    │       ├── mcqs_figure_3a.tiff
    │       ├── mcqs_figure_3b.tiff
    │       ├── mcqs_figure_3c.tiff
    │       ├── mcqs_figure_3d.tiff
    │       ├── mcqs_figure_3e.tiff
    │       ├── obisdian_figure_2a.png
    │       ├── obsidian_figure_2b.png
    │       ├── obsidian_figure_2c.png
    │       ├── obsidian_figure_3c.png
    │       ├── obsidian_figure_3c.tiff
    │       ├── obsidian_figure_3d.png
    │       ├── obsidian_figure_3d.tiff
    │       ├── obsidian_figure_3e.png
    │       ├── obsidian_figure_3e.tiff
    │       ├── obsidian_figure_5b_diverging.png
    │       ├── obsidian_figure_5b_upward_only.png
    │       ├── obsidian_figure_5c_diverging.png
    │       ├── obsidian_figure_5c_upward_only.png
    │       ├── obsidian_figure_5d.png
    │       ├── obsidian_figure_5e.png
    │       ├── obsidian_figure_5f.png
    │       ├── obsidian_figure_5g.png
    │       ├── obsidian_figure_s1a.png
    │       ├── obsidian_figure_s1b.png
    │       ├── obsidian_figure_s3.png
    │       ├── obsidian_figure_s4.png
    │       └── obsidian_figure_s5.png
    └── helpers
        ├── asserting_entries_format.ipynb
        ├── making_final_augmented_cns_dataset.ipynb
        ├── making_final_cns_dataset.ipynb
        ├── questions_for_human_eval.ipynb
        └── upload_checkpoint_to_hf.ipynb
```

