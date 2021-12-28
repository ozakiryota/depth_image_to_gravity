# depth_image_to_gravity
## Overview
This repository presents a deep neural network which estimates a gravity direction with a single shot depth image.
## Datasets
Some datasets are available at [ozakiryota/dataset_image_to_gravity](https://github.com/ozakiryota/dataset_image_to_gravity).
## Usage
The following commands are just an example.  
Some trained models are available in depth_image_to_gravity/keep.
### Regression
#### Training
```bash
$ cd ***/depth_image_to_gravity/docker/docker
$ ./run.sh
$ cd regression
$ python3 train.py
```
#### Inference
```bash
$ cd ***/depth_image_to_gravity/docker/docker
$ ./run.sh
$ cd regression
$ python3 infer.py
```
## Citation
If this repository helps your research, please cite the paper below.  
```TeX
@ARTICLE{ozaki2021,
	author = {Ryota Ozaki and Naoya Sugiura and Yoji Kuroda},
	title = {LiDAR DNN based self-attitude estimation with learning landscape regularities},
	journal = {ROBOMECH Journal},
	volume = {8},
	number = {26},
	pages = {10.1186/s40648-021-00213-5},
	year = {2021}
}
```
## Related repositories
- [ozakiryota/dataset_image_to_gravity](https://github.com/ozakiryota/dataset_image_to_gravity)
- [ozakiryota/dnn_attitude_estimation](https://github.com/ozakiryota/dnn_attitude_estimation)
