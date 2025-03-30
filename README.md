# DenseCLIP Segmentation on ADE20K

This repository contains an implementation of the segmentation task using DenseCLIP without relying on the `mmcv` module. The model is trained and evaluated on the ADE20K Challenge 2016 dataset.

## Features

- **DenseCLIP Integration**: Utilizes the DenseCLIP model for semantic segmentation.
- **No MMCV Dependency**: The implementation removes the need for `mmcv`, making it easier to run in environments with restricted installation permissions.
- **ADE20K Dataset**: Uses the ADE20K Challenge 2016 dataset for training and evaluation.
- **Custom Trainer**: Implements a trainer tailored for the segmentation task.

## Setup

### Environment Setup

Method: Conda environment

Python Version: 3.8

#### Creation Command:

```bash
conda create -n denseclip_pt17_py38 python=3.8 -y
conda activate denseclip_pt17_py38
```

**Reason for Older Python/PyTorch:** The target server system has GLIBC version 2.17. Newer PyTorch builds require GLIBC >= 2.27. PyTorch 1.7.1 with CUDA 11.0 was found to be compatible with the system's GLIBC.

### PyTorch and CUDA Installation

- **PyTorch Version:** 1.7.1
- **Torchvision Version:** 0.8.2
- **Torchaudio Version:** 0.7.2
- **CUDA Toolkit Version:** 11.0 (compatible with GLIBC 2.17 and Nvidia Driver 550.x supporting CUDA 12.4)

#### Installation Command:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -c conda-forge
```

### Additional Dependencies

These were installed using pip within the activated `denseclip_pt17_py38` environment:

```bash
pip install pyyaml timm==0.9.12 regex ftfy fvcore Pillow scikit-image tensorboard wget numpy==1.24.4 six matplotlib opencv-python
```

## Dataset Preparation

### Dataset: ADE20K ChallengeData 2016

#### Download and Extract:

```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

#### Storage Location:

```
~/DenseCLIP/
├── segmentation/
│   ├── ADEChallengeData2016/  <-- Extracted dataset HERE
│   │   ├── annotations/
│   │   ├── images/
│   │   └── ... (other dataset files)
│   ├── configs/
│   ├── datasets/
│   ├── denseclip/
│   ├── work_dirs/
│   └── train_denseclip.py
├── detection/
└── ...
```

**Note:** Do NOT add the dataset files to your Git repository. Ensure your `.gitignore` file includes entries like `segmentation/ADEChallengeData2016/` or `data/`.

### Configuration File Path Change

To ensure correct dataset usage, edit the configuration file: `~/DenseCLIP/segmentation/configs/denseclip_ade20k.yaml`.

Modify the `data` section:

```yaml
data:
  path: '.' # Point to the current directory (segmentation/)
  # ... rest of data config
```

Also, ensure all `pretrained:` keys are removed from this YAML file.

## Training

Run the training script:

```bash
python train_denseclip.py configs/denseclip_ade20k.yaml "--work-dir=work_dirs/pt17_run"
```

## Evaluation

Evaluate the model on the validation set:

```bash
python evaluate.py --dataset ADE20K --checkpoint path/to/checkpoint.pth
```

## Results

The segmentation model achieves competitive performance on ADE20K without requiring `mmcv`. Example segmented images are shown below:

*(Add sample segmentation results here)*

## Future Work

- **Enhancing Performance**: Experimenting with CoCoOp for improved generalization.
- **Additional Datasets**: Extending the implementation to other segmentation datasets.
- **Custom Prompt Learning**: Developing new prompt-based strategies for DenseCLIP.

## Citation

If you use this repository, please cite the original DenseCLIP paper:

```bibtex
@article{DenseCLIP,
  title={DenseCLIP: Extracting Dense Feature Representations from CLIP},
  author={Zhang, Haotian and Wu, Qirong and others},
  year={2023}
}
```

## Acknowledgments

- The ADE20K dataset: [MIT CSAIL](http://data.csail.mit.edu/places/ADEchallenge/)
- DenseCLIP: [GitHub Repository](https://github.com/muzairkhattak/multimodal-prompt-learning)


