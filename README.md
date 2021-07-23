# EAN: Event Adaptive Network

<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pan-towards-fast-action-recognition-via/action-recognition-in-videos-on-something-1)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something-1?p=pan-towards-fast-action-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pan-towards-fast-action-recognition-via/action-recognition-in-videos-on-something)](https://paperswithcode.com/sota/action-recognition-in-videos-on-something?p=pan-towards-fast-action-recognition-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pan-towards-fast-action-recognition-via/action-recognition-in-videos-on-jester)](https://paperswithcode.com/sota/action-recognition-in-videos-on-jester?p=pan-towards-fast-action-recognition-via) -->

PyTorch Implementation of paper:

> **EAN: Event Adaptive Network for Enhanced Action Recognition**
>
> Yuan Tian, Yichao Yan, Xiongkuo Min, Guo Lu, Guangtao Zhai, Guodong Guo, and Zhiyong Gao
>
> [[ArXiv](https://arxiv.org/abs/2107.10771)]

<!-- ## Updates -->
<!-- 
**[12 Aug 2020]** We have released the codebase and models of the PAN.  -->

## Main Contribution

Efficiently modeling spatial-temporal information in videos is crucial for action recognition.
In this paper, we propose a unified action recognition framework to investigate the **dynamic nature** of video content by introducing the following designs. First,
when extracting local cues, we generate the spatial-temporal
**kernels of dynamic-scale** to adaptively fit the diverse events.
Second, to accurately aggregate these cues into a global video
representation, we propose to mine the interactions only among
a few selected foreground objects by a Transformer, which yields
a sparse paradigm. We call the proposed framework as Event
Adaptive Network (EAN) because both key designs are adaptive
to the input video content. To exploit the short-term motions
within local segments, we propose a novel and efficient Latent
Motion Code (LMC) module, further improving the performance
of the framework.

<!-- 
<p align="center">
  <img src="https://user-images.githubusercontent.com/32992487/89706349-56563200-d997-11ea-8ed6-4ceca2883bad.gif" />
</p> -->

## Content

- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Pretrained Models](#pretrained-models)
  + [Something-Something-V1](#something-something-v1)
- [Testing](#testing)
- [Training](#training)
- [Other Info](#other-info)
  - [References](#references)
  - [Citation](#citation)
  - [Contact](#contact)

## Dependencies

Please make sure the following libraries are installed successfully:

- [PyTorch](https://pytorch.org/) >= 1.0
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)

## Data Preparation

Following the common practice, we need to first extract videos into frames for fast data loading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing. We have successfully trained on [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2), [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html) datasets with this codebase. Basically, the processing of video data can be summarized into 3 steps:

1. Extract frames from videos:

   * For Something-Something-V2 dataset, please use [data_process/vid2img_sthv2.py](data_process/vid2img_sthv2.py) 

   * For Kinetics dataset, please use [data_process/vid2img_kinetics.py](data_process/vid2img_kinetics.py) 

   * For Diving48 dataset, please use [data_process/extract_frames_diving48.py](data_process/extract_frames_diving48.py) 

2. Generate file lists needed for dataloader:

   * Each line of the list file will contain a tuple of (*extracted video frame folder name, video frame number, and video groundtruth class*). A list file looks like this:

     ```
     video_frame_folder 100 10
     video_2_frame_folder 150 31
     ...
     ```

   * Or you can use off-the-shelf tools provided by the repos: [data_process/gen_label_xxx.py](data_process/gen_label_xxx.py) 


3. Edit dataset config information in [datasets_video.py](datasets_video.py)


## Pretrained Models

Here, we provide the pretrained models of EAN models on Something-Something-V1 datasets. Recognizing actions in this dataset requires strong temporal modeling ability. EAN achieves state-of-the-art performance on these datasets. Notably, our method even surpasses optical flow based methods while with only RGB frames as input.

### Something-Something-V1

<div align="center">
<table>
<thead>
<tr>
<th align="center">Model</th>
<th align="center">Backbone</th>
<th align="center">FLOPs</th>
<th align="center">Val Top1</th>
<th align="center">Val Top5</th>
<th align="center">Checkpoints</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">EAN<sub>8F(RGB+LMC)</sub></td>
<td align="center" rowspan="3">ResNet-50</td>
<td align="center">37G</td>
<td align="center">53.4</td>
<td align="center">81.1</td>
<td align="center" rowspan="3">
[<a href="https://www.jianguoyun.com/p/DQ75LqkQ_vLOBhjx4IEE ">Jianguo Cloud</a>] 
<!-- or [<a href="https://share.weiyun.com/F2PJnUiE" rel="nofollow">Weiyun</a>]
To be release -->
</td>
</tr>
<tr>
<td align="center">EAN<sub>16(RGB+LMC)</sub></td>
<td align="center">74G</td>
<td align="center">54.7</td>
<td align="center">82.3</td>
</tr>
<tr>
<td align="center">EAN<sub>16+8(RGB+LMC)</sub></td>
<td align="center">111G</td>
<td align="center">57.2</td>
<td align="center">83.9</td>
</tr>
<!-- <tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">(46.6G+88.4G) * 2</td>
<td align="center">53.4</td>
<td align="center">81.1</td>
</tr> -->
</tbody>
</table>
</div>

## Testing 

For example, to test the EAN models on Something-Something-V1, you can first put the downloaded `.pth.tar` files into the "pretrained" folder and then run:

```bash
# test EAN model with 8frames clip
bash scripts/test/sthv1/RGB_LMC_8F.sh

# test EAN model with 16frames clip
bash scripts/test/sthv1/RGB_LMC_16F.sh

```

## Training 

We provided several scripts to train EAN with this repo, please refer to "[scripts](scripts/)" folder for more details. For example, to train PAN on Something-Something-V1, you can run:

```bash
# train EAN model with 8frames clip
bash scripts/train/sthv1/RGB_LMC_8F.sh

```

Notice that you should scale up the learning rate with batch size. For example, if you use a batch size of 32 you should set learning rate to 0.005.

## Other Info

### References

This repository is built upon the following baseline implementations for the action recognition task.

- [TSM](https://github.com/mit-han-lab/temporal-shift-module)
- [TSN](https://github.com/yjxiong/tsn-pytorch)

### Citation

Please **[★star]** this repo and **[cite]** the following arXiv paper if you feel our EAN useful to your research:

```
@misc{tian2021ean,
      title={EAN: Event Adaptive Network for Enhanced Action Recognition}, 
      author={Yuan Tian and Yichao Yan and Xiongkuo Min and Guo Lu and Guangtao Zhai and Guodong Guo and Zhiyong Gao},
      year={2021},
      eprint={2107.10771},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


### Contact

For any questions, please feel free to open an issue or contact:

```
Yuan Tian: ee_tianyuan@sjtu.edu.cn
```