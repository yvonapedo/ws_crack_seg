# [A Weakly Supervised Pavement Crack Segmentation Based on Adversarial Learning and Transformers](https://)

Authors: *Yvon Apedo*, *Huanjie Tao*

---
## Abstract

While pixel-level crack segmentation has demonstrated significant potential in infrastructure inspections, its reliance on detailed annotations poses challenges for widespread adoption due to the time-consuming and expensive nature of such labeling. In response, weakly supervised crack segmentation has garnered attention as it eliminates the need for pixel-level annotations. However, existing methods, primarily based on class activation maps (CAM), involve complex training processes and suffer from poor performance due to misalignment between CAM-generated labels and the true target in the image. Although convolutional neural networks (CNNs) have demonstrated outstanding performance in most semantic segmentation tasks, their limited receptive field poses a significant challenge in accurately segmenting road cracks. To address these challenges, we propose a weakly supervised approach for crack segmentation that simultaneously generates synthetic crack images and performs segmentation through adversarial learning. Unlike traditional methods that rely on coarse labels, our approach leverages synthetic crack images with corresponding labels, effectively eliminating issues of misalignment and noisy pseudo-labels. Our method introduces an encoder-decoder architecture for the segmentation model, incorporating a Transformer-based Feature Enhancement module in the encoder. This module is specifically designed to capture long-range dependencies and efficiently integrate both high- and low-level features. Additionally, the model includes a Hilo block to extract both high- and low-frequency information from the images, along with a Feature Shrinkage Decoder to aggregate and refine adjacent feature maps. Extensive experiments were conducted, and our model achieved an ODS of 67.12% on the CrackForest dataset, 60.11% on the Crack500 dataset, and 44.80% on the AEL dataset. These results clearly demonstrate that our model outperforms several existing weakly supervised pavement segmentation methods across these datasets.


---

## Usage
### Datasets
Download the CrackForest, CRACK500, AEL datasets and the file follows the following structure.

```
|-- datasets
    |-- crack500
        |-- trainA
        |   |--<pseudo_label.jpg>
        |--trainB
        |   |--<crack1.jpg>
        |--trainC
        |   |--<noncrack.jpg>
        |-- testA
        |   |--<pseudo_label.jpg>
        |--testB
        |   |--<crack1.jpg>
        |--testC
        |   |--<noncrack.jpg>
        ......
```

 
### Train

```
python train.py --dataroot ../datasets/crack500 --name check --model usseg --dataset_mode usseg  --display_env check --n_epochs 100 --lambda_A 10 --gpu_ids 0,1,2,3 --batch_size 4 --n_epochs_decay 100 --no_flip     
```
### Valid

```
python test.py
```

---
## Baseline Model Implementation


Our code heavily references the code in [SSVS](https://github.com/AISIGSJTU/SSVS).

---

