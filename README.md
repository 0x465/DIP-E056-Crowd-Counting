# DIP-E056-Crowd-Counting

This repo includes the implementation of CSRNet Crowd Counting using PyTorch.  
This project was developed using the repo of [leeyeehoo](https://github.com/leeyeehoo/CSRNet-pytorch) as a base.  

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
ShanghaiTech Dataset: [Kaggle](https://www.kaggle.com/tthien/shanghaitech-with-people-density-map)  

### Preprocessing Data  
1. Download/git clone repo.  
2. Download and unzip ShanghaiTech folder into ROOT. 

**NOTE**: please edit paths in scripts as needed before running.    

Root directory should have the following hierachy:       
**NOTE**: '*' files will only appear after running scipts.   
```
ROOT
  |-- shanghaitech_with_people_density_map
        |-- .....
  |-- out_image
        |-- ... *
  |-- out_video
        |-- ... *
  |-- images
        |--- images_to_be_counted
  |-- videos
        |--- videos_to_be_counted
  |-- json
        |-- part_A_test.json
        |-- part_A_train.json
        |-- part_A_train_with_val.json
        |-- part_A_val.json
        |-- part_B_test.json
        |-- part_B_train.json
        |-- part_B_train_with_val.json
        |-- part_B_val.json
  |-- output.avi *
  |-- CC.py
  |-- dataset.py
  |-- image.py
  |-- model.py
  |-- train.py
  |-- utils.py
  |-- partBmodel_best.pth.tar

```

### Crowd Counting  
Weights: 
[ShanghaiTech Part A](https://drive.google.com/file/d/1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH/view)   
[ShanghaiTech Part B](https://drive.google.com/file/d/1zKn6YlLW3Z9ocgPbP99oz7r2nC7_TBXK/view)   

To begin crowd counting, run the following python script with required arguments:   
- mode : {image, video, real_time}  
- path : {img_path, vid_path, device_id}  
  - file_path : full path to image/video folder or int value for videocapture device id   

*device_id* : 0 for webcam  
*video_type* : mp4, avi, etc

```python CC.py image img_path .../ROOT/images```   
```python CC.py video vid_path .../ROOT/videos/*.video_type```    
```python CC.py real_time device_id int```

### Citations

CSRNet
```
@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1091--1100},
  year={2018}
}
```
Please cite the Shanghai datasets and other works if you use them.
```
@InProceedings{Liu_2019_IROS,

author = {Liu, Weizhe and Lis, Krzysztof Maciej and Salzmann, Mathieu and Fua, Pascal},

title = {Geometric and Physical Constraints for Drone-Based Head Plane Crowd Density Estimation},

booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},

month = {November},

year = {2019}

}
```
