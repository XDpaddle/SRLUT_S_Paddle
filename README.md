## Practical Single-Image Super-Resolution Using Look-Up Table

[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/html/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.html) 

This repository contains an op-for-op Paddle reimplementation of [Practical Single-Image Super-Resolution Using Look-Up Table](https://openaccess.thecvf.com/content/CVPR2021/papers/Jo_Practical_Single-Image_Super-Resolution_Using_Look-Up_Table_CVPR_2021_paper.pdf).
It is modified from the original source code of Pytorch implementation(https://github.com/yhjo09/SR-LUT)


## Dependency
- Python 3.7
- Paddle 
- glob
- numpy
- pillow
- tqdm
- tensorboardx

## 1.Dataset
We use DIV2K dataset for training and Set5 dataset for testing

## 2. Testing
1. The testing image pairs is placed in ./val folder,you can change the image address in line 24 in val_model_S.py
   
2、The result image will be put in ./result/S/HR/

3、python val_model_S.py



## BibTeX
```
@InProceedings{jo2021practical,
   author = {Jo, Younghyun and Kim, Seon Joo},
   title = {Practical Single-Image Super-Resolution Using Look-Up Table},
   booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month = {June},
   year = {2021}
}
```

