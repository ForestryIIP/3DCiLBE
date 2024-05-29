# 3D-CiLBE : 3D City Long-term Biomass Estimating

This project repository is based on papers in Remote Sensing journals: ["Estimating Urban Forests Biomass with LiDAR by Using Deep Learning Foundation Models"](https://doi.org/10.3390/rs16091643) authored by Hanzhang Liu, Chao Mou*, Jiateng Yuan, Zhibo Chen, Liheng Zhong, and Xiaohui Cui. 

## 1. Data Prepare

The relevant data used by the model is available through an open platform.

- LiDAR : https://apps.nationalmap.gov/lidar-explorer/
- Single Tree Images (Point Cloud) : https://doi.org/10.25625/FOHUJM
- OSM : https://openmaptiles.org/

## 2. Structure

The code repository includes the following directory structure:

```bash
3DCiLBE_method/
├── LiDAR-SAM/ # Segmentation model is used to segment LiDAR data
├── MLiDAR-CLIP/ # A model for species identification of single trees
├── St-Informer/ # Time series prediction model
└── rangeprojection/ # Module for range projection of LiDAR data
```

## 3. Model Environment
#####  (1) LiDAR-SAM
- python>=3.8
- pytorch>=1.7
- torchvision>=0.8

##### (2) MLiDAR-CLIP
- pytorch=1.7.1
- cudatoolkit=11.0

##### (3) St-Informer
- python>=3.8
- cudatoolkit=11.0
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas == 0.25.1
- scikit_learn == 0.21.3
- torch == 1.8.0


## 4. Usage

To start with the training or running, use the following commands:

```bash
python train.py # Model training
python setup.py # Model running
```

## 5. CheckPoint
The model files are identified by name and only apply to the urban areas mentioned in the article.
- [Model Link](https://yunpan.bjfu.edu.cn:443/link/B4C15942C0ABAC82DB61526BACE3A308)
- sam_vitb : The model is suitable for completing range projection of LiDAR data and segmentation. Based on [OpenAI SAM](https://github.com/facebookresearch/segment-anything)
- sam_vitl : The model is suitable for completing range projection of LiDAR data and segmentation. Based on [OpenAI SAM](https://github.com/facebookresearch/segment-anything)
- mlidarclip10A ：The model is used to classify two-dimensional vegetation images. Model based on [CLIP RN50](https://github.com/OpenAI/CLIP) 
- mlidarclipRN50 : The model is used to classify two-dimensional vegetation images. Model based on [CLIP RN50*4](https://github.com/OpenAI/CLIP) 
- mlidarclipvit : The model is used to classify two-dimensional vegetation images. Model based on [CLIP ViT-B/32](https://github.com/OpenAI/CLIP) 

## 6. Citation

```
@article{Liu2024EstimatingUF,
  title={Estimating Urban Forests Biomass with LiDAR by Using Deep Learning Foundation Models},
  author={Hanzhang Liu and Chao Mou and Jiateng Yuan and Zhibo Chen and Liheng Zhong and Xiao-Ting Cui},
  journal={Remote Sensing},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:269704888}
}
```
