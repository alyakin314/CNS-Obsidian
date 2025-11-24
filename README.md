# CNS-Obsidian

**CNS-Obsidian: A Neurosurgical Vision-Language Model Built From Scientific Publications**

[![arXiv](https://img.shields.io/badge/arXiv-2502.19546-b31b1b.svg)](https://arxiv.org/abs/2502.19546)
[![HuggingFace](https://img.shields.io/badge/🤗-LLaVA--Next--Med--OLAB-yellow)](https://huggingface.co/NYU-OLAB/LLaVA-Next-Med-OLAB)

This repository contains the data processing and training code for CNS-Obsidian, a 34-billion parameter vision-language model specialized for neurosurgery. We demonstrate how domain-specific AI models can be built using curated, peer-reviewed scientific literature, establishing a transparent and reproducible framework for medical AI development.

## Overview

CNS-Obsidian was developed by fine-tuning [LLaVA-Next-34B](https://huggingface.co/liuhaotian/llava-v1.6-34b) on neurosurgical data extracted from 23,984 peer-reviewed articles, yielding 78,853 figures and 263,064 training samples. Through a three-stage curriculum training approach, the model achieved comparable real-world diagnostic performance comparable to GPT-4o in a blinded, randomized clinical trial while being orders of magnitude smaller and fully auditable.

**Key Contributions:**
- **Transparent Training Data:** Built entirely from peer-reviewed Neurosurgery Publications literature with explicit publisher permission
- **Novel Training Pipeline:** Three-stage curriculum (medical alignment → general medicine → neurosurgical specialization) with extensive ablation studies
- **Clinical Validation:** First blinded randomized trial of vision-language model chatbots in a clinical setting
- **Reproducible Framework:** Complete pipeline for converting scientific literature into vision-language training data

## Installation

To install CNS-Obsidian with all its dependencies (including PyTorch + CUDA 12.1 wheels) in a Python 3.12 environment, follow these steps:

### Create and activate a Python 3.12 environment:
```bash
conda create -n cns_obsidian python=3.12 -y
conda activate cns_obsidian
```

### Clone the repository and install with the extra index url for CUDA 12.1 wheels:
```bash
git clone git@github.com:alyakin314/CNS-Obsidian.git
cd CNS-Obsidian
pip install . --extra-index-url https://download.pytorch.org/whl/cu121 --editable
```

We use `--extra-index-url` so that PyTorch and its associated CUDA 12.1 wheels can be downloaded from the official PyTorch channel, while all other packages come from PyPI.

## Key Features

### 1. Data Processing Pipeline (`cns_obsidian/instruct/`)
Convert peer-reviewed figures and captions into three task-specific training formats:
- **Instruction Fine-Tuning (IFT):** Conversational question-answer pairs (127,076 samples)
- **Multiple-Choice Questions (MCQ):** Clinical vignettes with answer options (89,587 samples)
- **Differential Diagnosis (DDx):** One-line case summaries with tiered diagnoses (46,401 samples)

The pipeline uses GPT-4o and Claude Sonnet-3.5 with few-shot prompting to transform unstructured biomedical content into structured training data. See `notebooks/examples/data_processing_example_radiopaedia.ipynb` for a demonstration.

### 2. Three-Stage Curriculum Training
Based on the LLaVA-Med medical curriculum, we extended it with a neurosurgical specialization stage:

1. **Stage 1 – Medical Alignment** (`cns_obsidian/train/obsidian_stage_1_med_align.py`)
   - Freeze language model, train projection layers only
   - Data: 467K biomedical figure-caption pairs from PMC-15M
   - Duration: ~3.5 hours/epoch

2. **Stage 2 – General Medical IFT** (`cns_obsidian/train/obsidian_stage_2_med_finetune.py`)
   - Freeze vision model, train language model + projection layers
   - Data: 56K biomedical instruction-following conversations
   - Duration: ~30 minutes/epoch

3. **Stage 3 – Neurosurgical Specialization** (`cns_obsidian/train/obsidian_stage_3_cns_finetune.py`)
   - Freeze vision model, train language model + projection layers
   - Data: 263K neurosurgery-specific IFT, MCQ, and DDx samples
   - Duration: ~2 hours/epoch

Our final model configuration [5, 10, 10] denotes 5 epochs of Stage 1, 10 epochs of Stage 2, and 10 epochs of Stage 3. See ablation studies.

### 3. Dataset Visualization
Using [Nomic-Embed-Text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) embeddings with tSNE dimensionality reduction, we visualized the semantic space of our neurosurgical dataset compared to general biomedical literature. See `notebooks/figure_making/obsidian_figure_2.ipynb` for the data cartography analysis.

## Models

### Public Release
- **[LLaVA-Next-Med-OLAB](https://huggingface.co/NYU-OLAB/LLaVA-Next-Med-OLAB)** – Our recreation of LLaVA-Med using the LLaVA-Next-34B architecture. This intermediate checkpoint (Stage 1 + Stage 2 only) is publicly available and serves as a strong biomedical vision-language baseline.

### Private Models
- **CNS-Obsidian** Due to the proprietary nature of the Neurosurgery Publications data used in Stage 3 training, CNS-Obsidian weights are not publicly released. However, they can be made available to members of the Congress of Neurological Surgeons (CNS) upon request for research purposes. Contact the corresponding author for access.

## Results

### Benchmark Performance

| Model | GPT-Generated MCQs (n=1,282) | Claude-Generated MCQs (n=1,239) | CNS-SANS Questions (n=950) |
|-------|:----------------------------:|:-------------------------------:|:---------------------------:|
| LLaVA-Med (7B) | 42.74% | 29.12% | 28.74% |
| LLaVA-Next (34B) | 68.73% | 46.53% | 39.81% |
| LLaVA-Next-Med-OLAB (34B) | 68.96% | 53.70% | 43.98% |
| **CNS-Obsidian (34B)** | 79.18% | **74.39%** | 45.25% |
| GPT-4o | 81.16% | 64.48% | **65.60%** |
| Claude 3.5 Sonnet | **81.71%** | 63.92% | 56.20% |

### Clinical Trial Results
In a 92-day blinded randomized trial at NYU Langone Health (August 30 – November 30, 2024):
- **70 patient consultations** evaluated (32 CNS-Obsidian, 38 GPT-4o) from 959 total consults (7.3% utilization)
- **Diagnostic Helpfulness:** 40.62% (CNS-Obsidian) vs. 57.89% (GPT-4o), p=0.230
- **Diagnostic Accuracy:** 59.38% (CNS-Obsidian) vs. 65.79% (GPT-4o), p=0.626
- **Length-Adjusted Accuracy:** 16.88% (CNS-Obsidian) vs. 10.69% (GPT-4o), p=0.081

## Repository Structure

```
CNS-Obsidian
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

## License

This repository and its associated models can be subject to multiple licenses. The strictest license terms apply in all relevant cases:

- [NousResearch/Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B): Apache License 2.0
- [LLaVA-Next](https://huggingface.co/liuhaotian/llava-v1.6-34b): [Apache License 2.0](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/LICENSE)
- [LLaVA-Med Data](https://huggingface.co/microsoft/llava-med-7b-delta): CC BY NC 4.0
- [LLaVA-Med](https://huggingface.co/microsoft/llava-med-7b-delta): [Microsoft Research License Terms](https://github.com/microsoft/LLaVA-Med/blob/main/LICENSE)
- **Neurosurgery Publications Data:** Proprietary material from Wolters Kluwer, used with explicit permission. Restricted to internal research and evaluation only.
  - [General Terms of Use](https://www.wolterskluwer.com/en/terms-of-use)
  - [Permissions & Licensing](https://www.wolterskluwer.com/en/solutions/legal-regulatory/permissions-reprints-and-licensing)
  - [Licensing Requests](https://shop.lww.com/licensing)


## Contact

**Corresponding Author:** Anton Alyakin ([@alyakin314](https://github.com/alyakin314))  
Email: alyakin314@gmail.com  


## Citation

If you use CNS-Obsidian, LLaVA-Next-Med-OLAB, or any part of this codebase in your research, please cite our paper:

```bibtex
@misc{alyakin2025cnsobsidian,
      title={CNS-Obsidian: A Neurosurgical Vision-Language Model Built From Scientific Publications}, 
      author={Anton Alyakin and Jaden Stryker and Daniel Alexander Alber and Karl L. Sangwon and Jin Vivian Lee and Brandon Duderstadt and Akshay Save and David Kurland and Spencer Frome and Shrutika Singh and Jeff Zhang and Eunice Yang and Ki Yun Park and Cordelia Orillac and Aly A. Valliani and Sean Neifert and Albert Liu and Aneek Patel and Christopher Livia and Darryl Lau and Ilya Laufer and Peter A. Rozman and Eveline Teresa Hidalgo and Howard Riina and Rui Feng and Todd Hollon and Yindalon Aphinyanaphongs and John G. Golfinos and Laura Snyder and Eric Leuthardt and Douglas Kondziolka and Eric Karl Oermann},
      year={2025},
      eprint={2502.19546},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.19546}, 
}
```

