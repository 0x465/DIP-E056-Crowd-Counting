# DIP-E056-Crowd-Counting

This repo includes the implementation of Context Aware Crowd Counting.

### Prerequisites
Anaconda environment is strongly recommended.   

The following libraries was used:
- Python 3 
- OpenCV 3.4.2
- CUDA 10.2
- PyTorch 1.6.0
- Torchvision
- Scipy
- Matplotlib
- h5py

### Dataset
Refer to following pages:   
[ShanghaiTech Dataset](https://www.kaggle.com/tthien/shanghaitech-with-people-density-map)  
[grandcentral.avi](https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html)

### Preprocessing Data  
1. Download/git clone repo.  
2. Download and unzip ShanghaiTech folder into ROOT. 

Root directory should have the following hierachy:       
**NOTE**: '*' files will only appear after running scipts.   
```
ROOT
  |-- shanghaitech_with_people_density_map
        |-- .....
  |-- output
        |-- ... *
  |-- output.avi *
  |-- part_B_train.pth.tar
  |-- create_json.py
  |-- dataset.py
  |-- image.py
  |-- make_dataset.py
  |-- model.py
  |-- predict.py
  |-- predict_real_time.py
  |-- test.py
  |-- train.py
  |-- utils.py

```

### Prediction
**NOTE**: please edit paths in scripts as needed
Weights: 
[part_B_train.pth.tar](https://drive.google.com/file/d/15yHdpxYdcWO4NHuZz_Okri9BFlaAP1c7/view?usp=sharing)

Prediction can be done on:    
- new image using ```predict_image.py```  
- video uisng ```predict_video.py```   

### Citations
This project was developed using the repo of [weizheliu](https://github.com/weizheliu/Context-Aware-Crowd-Counting) as a base.  

```
@inproceedings{zhang2016single,
  title={Single-image crowd counting via multi-column convolutional neural network},
  author={Zhang, Yingying and Zhou, Desen and Chen, Siqin and Gao, Shenghua and Ma, Yi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={589--597},
  year={2016}
}
```
```
@InProceedings{Liu_2019_CVPR,

author = {Liu, Weizhe and Salzmann, Mathieu and Fua, Pascal},

title = {Context-Aware Crowd Counting},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

month = {June},

year = {2019}

}
```
```
@InProceedings{Liu_2019_IROS,

author = {Liu, Weizhe and Lis, Krzysztof Maciej and Salzmann, Mathieu and Fua, Pascal},

title = {Geometric and Physical Constraints for Drone-Based Head Plane Crowd Density Estimation},

booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},

month = {November},

year = {2019}

}
```
