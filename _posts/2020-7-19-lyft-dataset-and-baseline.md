---
title: "Lyft 车辆轨迹预测数据集及基线模型"
last_modified_at: 2020-06-21
categories:
  - Blog
tags:
  - self-driving
  - trajectory-prediction
  - weekly-update 
   
toc: true
toc_label: "目录"
toc_sticky: true
---

Level 5在6月25号公布了一个新的车辆轨迹预测数据集，相比于前段时间的包含320h数据的Argoverse，该数据集有1110h，并且其API也更容易使用，包含了完善的高清地图栅格化功能，从而可以较为容易地用CNN实现车辆轨迹预测。

# Level 5 车辆轨迹预测数据集

![Image for post](https://miro.medium.com/max/1200/1*ZhcLkG683pU1f0Xghow5vA.gif)

具体来说，Level 5 提供的数据集包含了以下几个部分：

- 17万个场景，每个场景大约是25秒的长度（大约1118小时）
- 1.6万英里的数据，由23个自动驾驶车辆收集
- 相关区域的高精地图，包括了道路结构、卫星地图、交通灯，人行横道等

下图可视化 展示了17万个场景中的一部分，可以发现场景还是很丰富的：

![image-20200719210222061]({{site.baseurl}}/assets/2020-07-19/image-20200719210222061.png)

另外，Level 5 预期将于2020年8月在kaggle上进行公开竞赛，也会进一步提高该领域的关注度，不过具体的要求还没有公布。

# ResNet 基线模型

Level 5还提供了一个完整的ResNet基线模型训练流程，并在论文中给出了相应的结果，其使用1s的过去车辆轨迹预测未来5s的ADE是2.28m，和当前的SOAT结果1.74m (预测单个轨迹）相差其实不大。考虑到该方法在实现上十分简单，在工程上还是有一定优势的。

![image-20200719204455360]({{site.baseurl}}/assets/2020-07-19/image-20200719204455360.png)

Level 5使用在ImageNet预训练的ResNet53作为backbone，然后替换掉了该模型的输入和输出，输入为N帧的过去车辆轨迹栅格化图像，单个图像的shape为 (W, H, 3)，N个图像在第三个维度垒起来，因此输入的shape实际为(W, H, 3 * N)，然后模型的最后一层被替换成了全连接层，输出的是（X * 10 * 2）维的向量，表示预测车辆的未来X秒轨迹。