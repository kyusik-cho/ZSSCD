<div align="center">
  
# Zero-Shot Scene Change Detection
### **AAAI 2025**

Kyusik Cho &emsp; Dong Yeop Kim &emsp; Euntai Kim  
Yonsei University

[![AAAI 2025](https://img.shields.io/badge/AAAI-2025-003973)](https://ojs.aaai.org/index.php/AAAI/article/view/32253)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2406.11210)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub Stars](https://img.shields.io/github/stars/kyusik-cho/ZSSCD?style=social)](https://github.com/kyusik-cho/ZSSCD)


</div>
<hr />

**TL;DR:** We present a novel, training-free approach to scene change detection by leveraging a tracking model.

<p align="center">
  <img src="assets/main.png" />
</p>


## Prerequisites

### Environmental Setup
```bash
conda env create -f environment.yml
conda activate zsscd
```

### Pretrained Weights
* Download the DEVA weights from [this repository](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) or use the bash command:

```bash
wget -P ./model_weights/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
```

* Download the SAM weights from [this repository](https://github.com/facebookresearch/segment-anything) or use the bash command:

```bash
wget -P ./model_weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

* Place the downloaded weights in the `model_weights` folder.



### Dataset Preparation 
Download the datasets and set the file structure as follows.  
Only the test set is required.  
Edit `dataset/path_config.py` to set the dataset directory path.

<details>
  <summary> Dataset folder structure (click to expand)</summary>

* [ChangeSim](https://github.com/SAMMiCA/ChangeSim)
```bash
ChangeSim
├── Query
│   └── Query_Seq_Test
│       ├── Warehouse_6
│       │   ├── Seq_0
│       │   │   └── ...
│       │   └── Seq_1
│       │       └── ...
│       ├──  ...
│       │
│       └── Warehouse_8
│           ├── Seq_0
│           │   └── ...
│           └── Seq_1
│               └── ...
│
└── Ref
    └── Ref_Seq_Test
        ├── Warehouse_6
        │   ├── Seq_0
        │   │   └── ...
        │   └── Seq_1
        │       └── ...
        ├──  ...
        │
        └── Warehouse_8
            ├── Seq_0
            │   └── ...
            └── Seq_1
                └── ...
```

* VL-CMU-CD
```bash
VL-CMU-CD-binary255
└── test
    ├── t0
    ├── t1
    └── mask
```


* [PCD](https://kensakurada.github.io/pcd_dataset.html)
```bash
PCD
├── TSUNAMI
│   ├── t0
│   ├── t1
│   ├── ground_truth
│   └── mask
└── GSV
    ├── t0
    ├── t1
    ├── ground_truth
    └── mask
```

</details>

## Evaluation
* ChangeSim
```bash
$ python main.py
$ python main.py --changesim_subset dark
$ python main.py --changesim_subset dust
```

* VL_CMU_CD
```bash
$ python main.py --dataset VL_CMU_CD
```

* PCD
```bash
$ python main.py --dataset TSUNAMI
$ python main.py --dataset GSV
```

## Acknowledgements
Our PyTorch-based implementation is based on the following projects and repos.
* [Segment Anything](https://github.com/facebookresearch/segment-anything)
* [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)
* [GCDSS](https://github.com/JethroPeng/GCDSS)


## Cite
Please cite our paper if our work is helpful to your research:
```
@inproceedings{cho2025zero,
  title={Zero-shot scene change detection},
  author={Cho, Kyusik and Kim, Dong Yeop and Kim, Euntai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2509--2517},
  year={2025}
}
```

