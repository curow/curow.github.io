---
title: "Argoverse 数据集详解"
last_modified_at: 2020-05-24
categories:
  - Blog
tags:
  - self-driving
  - trajectory-prediction
  - weekly-update
  - dataset

toc: true
toc_label: "目录"
toc_sticky: true
---

Argoverse数据集介绍

## 数据集

Argoverse[^1]的数据集包括3D Tracing以及Motion Forecasting这两大类，我们关注的是Argoverse的Motion Forecasting这个数据集，包含了从超过1000个驾驶小时的数据中提取的327,790条有价值的场景，每个场景包含了自动驾驶车辆5秒钟的行驶轨迹，同时跟踪所有其他参与者（例如汽车，行人）。 并将它们分为208272个训练序列，40127个验证序列和79391个测试序列。

这里每个场景都包含以10 Hz采样的所有跟踪对象的2D鸟瞰图。如下图所示，这是某一个场景的静态图示，绿色的是自动驾驶车辆，红色的是某一个被跟踪的车辆，淡蓝色的点是其他被跟踪车辆。

![motion_forecasting_still]({{site.baseurl}}/assets/2020-05-24/motion_forecasting_still.jpg)

## 地图

除了车辆轨迹数据集，Argoverse还给出了行驶地区的高精地图，分为三个部分：

### Vector Map: Lane-Level Geometry

![vector-map-centerlines]({{site.baseurl}}/assets/2020-05-24/vector-map-centerlines.jpg)

如图所示，给出了环境中车道级别的精确信息。

### Rasterized Map: Drivable Area

![driveable-area]({{site.baseurl}}/assets/2020-05-24/driveable-area.jpg)

同时给出了一米精度的可驾驶区域binary map。

### Rasterized Map: Ground Height

![ground-surface]({{site.baseurl}}/assets/2020-05-24/ground-surface.jpg)

另外也给出了这些区域的高度图。

## 竞赛

### 输入输出

Argoverse预测竞赛是给定待预测车辆的20帧（2s）过去信息，预测这些车辆的未来30帧（3s）信息，因为Argoverse认为对于车辆来说，5s已经足够捕获到车辆轨迹信息最关键的部分了（如换道或者通过交叉口），另外，预测任务可以使用social context以及map information。

### 评估指标

预测的指标主要是**minimum Average Displacement Error（minADE）**以及**minimum Final Displacement Error（minFDE）**over K predictions，其中K = 1， 3， 6，9。值得注意的细节是minADE是指具有minFDE的预测轨迹的ADE。

另外，考虑到minADE和minFDE只能评估最好的一个轨迹，例如如果一个预测器给出了1个最好的轨迹和4个很差的轨迹，那么它就会优于给出5个很好的轨迹的预测器。所以Argoverse还采用了另外两个评测指标，一个是**Drivable Area Compliance （DAC）**，即如果一个预测器给出m个轨迹，而其中n条在某个时刻离开了可驾驶地区，那么DAC就等于 (n - m) / n。还有一个是**Miss Rate （MR）**，和DAC类似考察的是多数轨迹的优劣，如果m条轨迹中n条的FDE大于2m，那么MR就等于m/n。

总的来说，我们想要minADE、minFDE、MR越小越好，DAC越大越好。

下图是来自于最新Argoverse Motion Forecasting竞赛的结果, 可以看到在预测单一轨迹的时候，最好的方法给出的minADE是2m左右，minFDE是4m左右。而在预测6条轨迹的时候，minADE为0.9m，minFDE为1.5m。

另外这里相对于原论文还多了**Probabilistic minimum Average Displacement Error (p-minADE)**，这个是optional的，如果模型给出了每条轨迹的概率的话就可以用这个来计算。

![argoverse-motion-forecasting-competition-leaderboard]({{site.baseurl}}/assets/2020-05-24/argoverse-motion-forecasting-competition-leaderboard.png)

下面几个图是Argoverse竞赛输入输出的例子，橙色的轨迹代表某个跟踪车辆在最初2秒钟内的运动，绿色的轨迹代表我们的前k个预测轨迹，红色的轨迹代表的是真实轨迹。

![img]({{site.baseurl}}/assets/2020-05-24/motion_forecasting_merge.png)

![img]({{site.baseurl}}/assets/2020-05-24/motion_forecasting_right.png)

![img]({{site.baseurl}}/assets/2020-05-24/motion_forecasting_split.png)

![img]({{site.baseurl}}/assets/2020-05-24/motion_forecasting_swirl.png)

## 参考链接

[^1]: https://www.argoverse.org/data.html