---
title: "Study on Self-Driving Vehicle Sensor Range"
categories:
  - Blog
tags:
  - self-driving
  - weekly-update

toc: true
toc_label: "目录"
toc_sticky: true
---

这篇文章的目的是明确主流无人驾驶车辆车载传感器的感知范围，选取了Waymo、Tesla、Baidu Apollo这几个头部的无人驾驶公司来分析。三者在技术路线以及经营模式上有着较大的差距，有助于我们较为全面地了解目前典型的自动驾驶车辆只依靠单车传感器的感知范围详细数值。

# 总体

![img]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/Autonomous+Vehicle+Sensors+1024px.jpg)

上图中总结了一辆无人驾驶车辆可能用到的各种车载传感器，分别是：

- LIDAR，激光雷达
- GNSS，全球定位系统
- IMU，惯性传感器
- Cameras，摄像头
- Themal Cameras，红外摄像头
- Long-range RADAR，长距毫米波雷达
- Short-/Medium-range RADAR，短/中距毫米波雷达
- Ultrasound，声呐

在感知方面，我们需要关注的主要是摄像头、激光雷达、毫米波雷达、以及声呐。下图比较形象地展示了各个传感器的感知范围，但是这个感知范围并不是按照比例尺来做的，所以只能作为直观的理解。

![img]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/Autonomous-Vehicle-Sensors-02-790px.png)

下表是一份2020年的技术报告[^1]中总结的各个传感器有效范围、价格、以及每秒输出数据量，以上的传感器图示也是来自于这个报告。

| Sensor     | Measurement distance (m) | Cost  ($)    | Data rate (Mbps) |
| ---------- | ------------------------ | ------------ | ---------------- |
| Cameras    | 0-250                    | 4–200        | 500-3500         |
| Ultrasound | 0.02-10                  | 30-400       | < 0.01           |
| RADAR      | 0.2-300                  | 30-400       | 0.1-15           |
| LIDAR      | Up to 250                | 1,000-75,000 | 20-100           |

可以看到这些传感器可测量的有效范围能达到数百米。不过不同的公司采用的传感器方案也有较大的区别，所以下面对三家的具体感知范围分别做一个介绍。

# Waymo

![image-20200503070301321]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200503070301321.png)

![image-20200503070513571]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200503070513571.png)

以上两图分别是Waymo Driver的传感器方案图以及其实时运行时候的UI，它使用了360度LIDAR探测道路目标，使用9个视觉摄像头跟踪道路，并使用RADAR识别汽车附近的障碍物。按Waymo的最新报道[^2]，其第五代自动驾驶系统LIDAR可感知范围在300m以上，视野为360度，提供了车辆驾驶时的鸟瞰视角，而它的长距摄像头以及360度视觉系统可以感知高于500米的重要特征（如行人、交通灯等）。

# Tesla

![image-20200503070206251]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200503070206251.png)

![image-20200503071251539]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200503071251539.png)

![Pin on logistics福田物流]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/5e43939410133f4dd48e0d5d41c24544.jpg)

Tesla采用的是视觉摄像头和RADAR的组合，没有采用LIDAR，但是感知范围也可以达到160m[^3]，从上图也可以看到它不仅能感知到周围车辆，对更前方的车辆也能有一个较好的识别。

# Baidu Apollo

![image-20200503070357148]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200503070357148.png)

![image-20200503071325739]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200503071325739.png)

![image-20200503070851850]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200503070851850.png)

![image-20200502203330296]({{site.baseurl}}/assets/2020-05-03-self-driving-vehicle-sense-range/image-20200502203330296.png)

Baidu Apollo的方案[^4]和Waymo其实是相似的，不过相比Waymo的客制化硬件，Apollo选择了联合其他厂商，所以并没有一个固定的传感器搭配，上面展示的是他在长沙试运营的Robotaxi，能见度同样可以达到300m以上，并且是360度视角。

# 总结

综上所述，我认为在周围车辆轨迹预测方面，自动车辆本身的传感器数据已经能提供相当丰富的信息。自动驾驶领域的V2X技术可能在更上层，如全局路径规划或者拥塞控制方面，会有更大的应用空间。

# 参考链接

[^1]: https://www.wevolver.com/article/2020.autonomous.vehicle.technology.report
[^2]: https://blog.waymo.com/2020/03/introducing-5th-generation-waymo-driver.html
[^3]: https://www.tesla.com
[^4]: http://apollo.auto/robotaxi/index_cn.html

