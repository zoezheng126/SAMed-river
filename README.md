# Segment Any Stream: Scalable Water Extent Detection with the Segment Anything Model
[[`Paper`](https://openreview.net/forum?id=BaZZzH7EgA)] [[`Poster`](https://openreview.net/pdf?id=BaZZzH7EgA)]\

The code of SAS is build upon [SAM](https://github.com/facebookresearch/segment-anything) and [SAMed](https://github.com/hitachinsk/SAMed), and we sincerely appreciate the contributions of the creators of these projects.\
If you find the code or the data useful, please cite our paper:
```
@inproceedings{zheng2023segment,
  title={Segment Any Stream: Scalable Water Extent Detection with the Segment Anything Model},
  author={Zheng, Haozhen and Zhang, Chenhui and Guan, Kaiyu and Deng, Yawen and Wang, Sherrie and Rhoads, Bruce L and Margenot, Andrew J and Zhou, Shengnan and Wang, Sheng},
  booktitle={NeurIPS 2023 Computational Sustainability: Promises and Pitfalls from Theory to Deployment},
  year={2023}
}
```

# Environment Setup
Please refer to [SAM](https://github.com/facebookresearch/segment-anything) for basic setup.

# Quick Start
* git clone this repo \
```git clone https://github.com/zoezheng126/SAMed-river.git```
* checkout to development branch \
```git checkout development```
* conda activate SAMed (conda init with environment.yml) \
```conda env create -f environment.yml```
```conda activate SAMed_test```
* Request SAMed_input dataset from Zoe (zoezheng126@hotmail.com) or download from [google drive](https://drive.google.com/drive/folders/1I36LyUu1Ad1rmoFbcPCDzz3QKWsmqUGD?usp=drive_link)
* Move `SAMed_input` directory to the current directory
* Download `sam_vit_b_01ec64.pth` and `LoRA8.pth` to `checkpoints` folder from [Google Drive](https://drive.google.com/drive/folders/16L5es291O221JxK5KmYK9pg55dnm9UKH?usp=sharing)
* Run `bash prepareDataset.sh .datasets/test SAMed_input/test_image SAMed_input/test_mask` to store original dataset into numpy form at `.datasets/test`
* Run `bash rivertest.sh` to get test results in `outputs/LoRA_test`. Estimated time 5 mins using one Delta A40 node 

# Training
Read TIFF images and shapefile masks during data loading:
* Run `bash rivertrain.sh` 
* In another terminal, run `tensorboard --log-dir <model_output/your-train-session-name/log> --host your-ip-address` (please read tensorboard doc for implementation) 
* After training, you can copy the new checkout point to checkpoints folder and update the bash command in `rivertest.sh`

Or preload the data into .npy files and train afterward, please read `rivertrain.sh` for more details:
* Run `bash prepareDataset.sh .datasets/train SAMed_input/train_image SAMed_input/train_mask` to store original dataset into numpy form at `.datasets/train`
* start training `bash rivertrain.sh`

# Citation
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
@article{samed,
  title={Customized Segment Anything Model for Medical Image Segmentation},
  author={Kaidong Zhang, and Dong Liu},
  journal={arXiv preprint arXiv:2304.13785},
  year={2023}
}
```



