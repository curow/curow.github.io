---
title: "Survey on Vehicle Trajectory Prediction"
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
这篇文章主要整理了这段时间所看轨迹预测方面论文的大致思路，也整理了相关的数据集和性能指标。

# 目的

预测周围车辆的轨迹，侧重点在于换道行为的预测。

# 研究动机总结

1. 对换道意图的准确预测有助于提高驾驶安全保护系统的有效性。

   High precision prediction of lane changing intent is helpful to enhance proactivity in driving safety protection.

2. 有助于避免潜在的意外事件发生，帮助无人驾驶车辆更好地做出决策。

   help to avoid potential accidents and make the best decision to ensure safety and comfort

3. 据报道，车辆碰撞主要是危险的换道行为引起的。

   Currently, there are aggressive drivers conducting a risky lane change even when sufficient speed and distance are not guaranteed. It was reported that car crashes mainly occur because of lane changing.

# 思路总结

一个效果可以达到state-of-the-art的模型可能需要考虑以下几点：

1. interaction-aware: 意识到周围的车辆轨迹和ego vehicle自身的轨迹是相互影响的，因此在预测周围车辆轨迹的时候可能也需要考虑到ego vehicle的intention是什么。
2. physical-aware：周围的环境对车辆轨迹也有较大的影响，知道自己在什么场景下对预测同样是有帮助的。
3. traffic-rules-aware：车辆的轨迹一般是遵循交通规则的，知道这方面的信息对轨迹预测肯定也是有帮助的，比如可以排除一些预测出的不遵守交通规则的轨迹。

问题其实在于如何将这些信息整合在模型中，如何让模型能够通过数据来自动学习这些东西（交通规则可能需要事先指定，但是对交通规则的遵循程度是应该通过数据学习的）。

# 文献阅读

## 自己查的

1. Multi-Modal Trajectory Prediction of Surrounding Vehicles withManeuver based LSTMs

   N Deo, MM Trivedi - 2018 IEEE Intelligent Vehicles Symposium …, 2018 - ieeexplore.ieee.org

   这篇文章是用LSTM预测ego vehicle本身的轨迹，遵循的思路是先根据ego vehicle以及周围的车辆轨迹判断当前车辆的maneuver类别，然后再根据不同的类别预测某个类别下的轨迹。

   

2. Sophie: An attentive gan for predicting paths compliant to social and physical constraints

   A Sadeghian, V Kosaraju… - Proceedings of the …, 2019 - openaccess.thecvf.com

   行人轨迹预测，利用了场景的鸟瞰图（top-down view）以及场景中agents的过去轨迹，来预测他们的未来轨迹。这篇结合了LSTM encoder decoder，以及attention机制（包括physical attention以及social attention）。尽管不是车辆轨迹预测，但是我认为还是比较有参考价值，因为data-driven的方法普适性要更强，这个领域用到的方法完全可以迁移到车辆轨迹预测中来。

   

3. Multiple futures prediction

   C Tang, RR Salakhutdinov - Advances in Neural Information …, 2019 - papers.nips.cc

   这篇文章直接预测某个场景下多个车辆的未来可能的多个轨迹，不仅考虑了场景中agents之间的互动，还考虑了scene context，不同的agents是不同的RNN网络来预测，但是它们的权重是共享的。

## 老师推荐（ref1：24篇，ref2：54篇）

### ref2

1. Key feature selection and risk prediction for lane-changing behaviors based on vehicles' trajectory data

   T Chen, X Shi, YD Wong - Accident Analysis & Prevention, 2019 - Elsevier

   这个用的是传统的机器学习方法（fault tree analysis以及k-means clustering）来对车辆变道风险程度进行分级的，和我们的问题关系不大。

   

2. Probabilistic Trajectory Prediction for Autonomous Vehicles with Attentive Recurrent Neural Process

   J Zhu, S Qin, W Wang, D Zhao - arXiv preprint arXiv:1910.08102, 2019 - arxiv.org 

   根据某一辆车自身以及周围车辆的轨迹，预测这辆车的换道意图以及轨迹。用的大致是LSTM encoder-decoder以及attention机制的结合，但是文章里被叫做attentive recurrent neural process(ARNP)

   

3. Interactive trajectory prediction for autonomous driving via recurrent meta induction neural network

   C Dong, Y Chen, JM Dolan - 2019 International Conference on ..., 2019 - ieeexplore.ieee.org

   这篇也用到了一种Neural Process，叫做Conditional Neural Process（CNP）。算法叫做Recurrent Meta Induction Network，建立在CNP的基础上，但是为了满足时序性，sub-net使用了LSTM。目的和用到的数据和上一篇基本一样。

   

4. Simultaneous modeling of car-following and lane- changing behaviors using deep learning

   X Zhang, J Sun, X Qi, J Sun - Transportation research part C: emerging ..., 2019 - Elsevier

   还是根据某一辆车和以及以及其附近车辆的轨迹，预测这辆车的轨迹。用的就是LSTM，只是训练上用的是自己提出的hybrid retraining constrained method(HRC)，有点类似半监督学习。

   

5. A data-driven lane-changing model based on deep learning

   DF Xie, ZZ Fang, B Jia, Z He - Transportation research part C: emerging ..., 2019 - Elsevier

   将换道行为建模分成了两部分，先是用Deep belief network (DBN)来做换道决策，然后用LSTM生成换道轨迹。

   

6. Interactive Trajectory Prediction of Surrounding Road Users for Autonomous Driving Using Structural-LSTM Network

   L Hou, L Xin, SE Li, B Cheng... - IEEE Transactions on ..., 2019 - ieeexplore.ieee.org

   根据某一辆车和以及以及其附近车辆的轨迹，预测所有这些车的轨迹。用到的是他们设计的Structural-LSTM，基本思想是输入每个车辆对应的LSTM单元的不仅是这个车本身这个时刻的信息，还有它们邻近车辆的信息。

   

7. Intention-aware Lane Changing Assistance Strategy Basing on Traffic Situation Assessment

   J Wu, S Liu, R He, B Sun - 2020 - sae.org

   这篇是SAE的，无法下载。

   

8. Short-term prediction of safety and operation impacts of lane changes in oscillations with empirical vehicle trajectories

   M Li, Z Li, C Xu, T Liu - Accident Analysis & Prevention, 2020 - Elsevier

   这篇是分析换道对交通流的影响，没太大关系。

   

9. A hierarchical prediction model for lane-changes based on combination of fuzzy C-means and adaptive neural network

   J Tang, S Yu, F Liu, X Chen, H Huang - Expert systems with applications, 2019 - Elsevier

   无监督学习和监督学习相结合来分析数据集，预测ego vehicle的换道意图。

   

10. A Dual Learning Model for Vehicle Trajectory Prediction
    M Khakzar, A Rakotonirainy, A Bond... - IEEE Access, 2020 - ieeexplore.ieee.org

    这篇仍然是预测ego vehicle的轨迹，结构是LSTM encoder decoder，输入的是ego vehicle周围的Lane Occupancy Map以及Risk Map，Risk是拿TTC（Time To Collision）来衡量的。

   

11. Multi-modal vehicle trajectory prediction based on mutual information

    C Fei, X He, X Ji - IET Intelligent Transport Systems, 2019 - IET

    预测ego vehicle的轨迹，LSTM encoder decoder，输入信息是ego vehicle以及周围6辆车辆的轨迹，结构上的不同是Muti-modal decoder，也就是根据输入先判断ego vehicle想要干什么（lane keeping，lane changing  left，lane changing right），然后根据intention的不同给3个不同的LSTM网络分别预测轨迹。

    

12. A Probabilistic Framework for Trajectory Prediction in Traffic Utilizing Driver Characterization

    JS Gill, P Pisu, MJ Schmid - 2019 IEEE 2nd Connected and ..., 2019 - ieeexplore.ieee.org

    这篇提出了一个抽象的概率框架来预测某个车辆的轨迹，它将某个drive做决策的过程分成4个部分来考虑：intent determination, maneuver preparation, gap acceptance, maneuver execution，并没有学习的部分，也没有实验。

    

13. Lane Attention: Predicting Vehicles' Moving Trajectories by Learning Their Attention over Lanes

    J Pan, H Sun, K Xu, Y Jiang, X Xiao, J Hu... - arXiv preprint arXiv ..., 2019 - arxiv.org

    这是百度的一篇，目的是预测某个车辆的轨迹，照常还是有LSTM，比较新奇的是他的建模里面并不考虑周围车辆的影响，而是把侧重点放在车辆周围的车道上，并且对车道的位置确定用的是非欧的坐标系，比较类似Frenet坐标，然后和车道之间的关系是用类似图神经网络（GNN）来建模的。

    

14. Probabilistic intention prediction and trajectory generation based on dynamic bayesian networks

    G He, X Li, Y Lv, B Gao, H Chen - 2019 Chinese Automation ..., 2019 - ieeexplore.ieee.org

    用的是Dynamic Bayesian Networks（基于概率的方法）来预测ego vehicle的轨迹，利用ego vehicle和周围车辆的距离来判断ego vehicle的intention。

    

15. PiP: Planning-informed Trajectory Prediction for Autonomous Driving

    H Song, W Ding, Y Chen, S Shen, MY Wang... - arXiv preprint arXiv ..., 2020 - arxiv.org

    这篇和我们的目标相关性较大，研究的是如何预测ego vehicle周围车辆的轨迹，除了用到这些车辆过去的轨迹信息之外，还用到了ego vehicle计划的轨迹，因为在一些互动性很强的场景下（如汇入）ego vehicle本身的轨迹也会对周围车辆的轨迹产生较大的影响，在预测其他车辆轨迹的时候考虑到自己的planning对这些车辆的影响是比较重要的。

    

16. Advanced Adaptive Cruise Control Based on Operation Characteristic Estimation and Trajectory Prediction

    H Woo, H Madokoro, K Sato, Y Tamura, A Yamashita... - Applied Sciences, 2019 - mdpi.com

    这篇文章出发点和我们的是基本一致的，不过他不仅想要对周围车辆轨迹做出预测，还会根据做出的预测（碰撞风险）来规划自己的纵向行为（加速减速）。预测分成3个部分，首先是周围车辆的intention预测（变道，保持，调整，到达）以及用来预测各个车辆的驾驶模型参数调整，然后根据GM模型来预测这些车辆未来的轨迹，intention预测是用的SVM，参数调整用的是Levenberg-Marquardt算法。总的来说这篇文章提出的还是基于模型的算法。

    

17. Vehicle trajectory prediction using recurrent LSTM neural networks

    A Bükk, R Johansson - 2020 - odr.chalmers.se

    这是一篇硕士论文，用LSTM来预测ego vehicle的轨迹，没用到周围车辆的轨迹信息，只用了自己的，参考价值应该不大。

    

18. Deep Predictive Autonomous Driving Using Multi-Agent Joint Trajectory Prediction and Traffic Rules

    K Cho, T Ha, G Lee, S Oh - ... on Intelligent Robots and Systems (IROS ..., 2019 -rllab.snu.ac.kr

    这篇文章提出了自动驾驶车辆的实现框架，其中也包含了对自己以及周围车辆轨迹的预测，比较新奇的是它这里考虑了对交通规则的遵守情况，即预测的轨迹需要一定程度上符合交通规则。

    

19. A hybrid model for lane change prediction with V2X-based driver assistance

    T Xu, R Jiang, C Wen, M Liu, J Zhou - Physica A: Statistical Mechanics and ..., 2019 -Elsevier

    这篇文章利用的是隐含马尔科夫模型（HMM）以及高斯混合模型（GMM）来预测ego vehicle周围车辆的换道决策，只预测他们是否换道，不涉及轨迹的预测，周围车辆的信息是通过V2X获取的。

    

20. Attention Based Vehicle Trajectory Prediction

    K Messaoud, I Yahiaoui, A Verroust-Blondet... - IEEE Transactions on ..., 2020 - hal.inria.fr

    这一篇基本和标题一样，用的是LSTM加上attention机制，预测ego vehicle本身的轨迹，而且是在一个比较大的尺度上考虑的，因此和周围车辆基本没有什么关系，参考价值不大。

    

21. Gauss mixture hidden Markov model to characterise and model discretionary lane-change behaviours for autonomous vehicles

    H Jin, C Duan, Y Liu, P Lu - IET Intelligent Transport Systems, 2020 - IET

    无法下载
    
    
    
22. Learning Probabilistic Intersection Traffic Models for Trajectory Prediction

    A Patterson, A Gahlawat, N Hovakimyan - arXiv preprint arXiv:2002.01965, 2020 - arxiv.org

    这篇文章采用高斯过程对单一车辆的intent（左转，右转，直行）进行预测，预测时需要的信息为这个车辆本身的轨迹。模型训练使用的是模拟数据集，模拟某一车辆在交叉路口如何行驶，然后对数据集进行聚类，分别训练对应intent的模型。在运行时，根据3种intent模型和目前车辆轨迹的距离判断车辆到底是要左转，右转，还是直行。

    

23. Vehicle Trajectory Prediction Using Intention-based Conditional Variational Autoencoder

    X Feng, Z Cen, J Hu, Y Zhang - 2019 IEEE Intelligent ..., 2019 - ieeexplore.ieee.org

    还是预测单一车辆的未来轨迹，输入是车辆自身和周围其他车辆的轨迹，对seq to seq的模型进行了修改，增加了intent判断的分支，另外还采用了深度生成模型 CVAE（条件变分自编码器）来根据intent来生成车辆未来轨迹。

    

24. Lane change identification and prediction with roadside LiDAR data

    Y Cui, J Wu, H Xu, A Wang - Optics & Laser Technology, 2020 - Elsevier

    在路边安装LiDAR，通过LiDAR数据来提取道路信息、追踪车辆轨迹信息，实时预测车辆的变道行为，并通过RSU来广播车辆变道警告。他这里变道行为的预测是基于规则的，主要判断依据是车辆的横向偏移（和最近车道线的横向距离变换）。由于只预测变道还是不变道，所以他用的指标是PA（Prediction Accuracy，所有车辆中预测正确的比例）以及PT（Prediction Time，车辆穿过中心线的时间减去预测到车辆会变道的时间）。

    

25. Vehicle trajectory prediction algorithm in vehicular network

    L Wang, Z Chen, J Wu - Wireless Networks, 2019 - Springer

    这篇文章是在车联网的背景下讨论车辆轨迹预测的，提出了一个利用车辆网储存和分析车辆轨迹信息的框架，然后用GMM（Gaussian Mixed Model）来建模车辆轨迹，用GPR（Gaussian Process Regression）来预测车辆的未来轨迹，用的都是车辆的GPS数据。

    

26. Manifold Learning for Lane-changing Behavior Recognition in Urban Traffic

    J Li, C Lu, Y Xu, Z Zhang, J Gong... - 2019 IEEE Intelligent ..., 2019 - ieeexplore.ieee.org

    主要思想是采用流形学习（manifold learning，主要思想是将高维数据降维到低维度，从而反映数据的本质特征）降维，从而区分邻近车辆的三个状态（换道前，正在换道，换道后）。这里用到的数据是车辆前置摄像头的视频数据，具体流程是先用YOLOv3来提取车辆的边框，并获取ego vehicle的车辆行驶状态，用PCA和ISOMAP（均是典型的降维算法）来对数据进行降维，最后用SVM来区分。

    

27. A Cooperative Driving Strategy Based on Velocity Prediction for Connected Vehicles with Robust Path-following Control

    Y Chen, C Lu, W Chu - IEEE Internet of Things Journal, 2020 - ieeexplore.ieee.org

    这篇文章提出的是一个协同自动驾驶的整体方案，包括了前车速度预测（假设其不变道，通过V2V获取前车信息）、运动规划、轨迹控制这3大方面，其中速度预测是用vallina LSTM，参考价值不大。

    

28. Autonomous Driving using Safe Reinforcement Learning by Incorporating a Regret-based Human Lane- Changing Decision Model

    D Chen, L Jiang, Y Wang, Z Li - arXiv preprint arXiv:1910.04803, 2019 - arxiv.org

    提出了一个考虑到周围车辆变道的自动驾驶框架，主要有两个部分，一个是采用Regret Theory来建模和预测周围车辆的换道意图，另一个是采用SafeRL的框架来将对周围车辆换道意图的决策融入到模型的训练上，从而使得RL模型在训练和测试的时候碰撞率为0%，相比于传统的RL模型8.7%的碰撞率要安全很多。

    换道意图模型主要思想是考虑换道和车道保持的costs，即换道可能使得车辆能更快地行驶，也可能发生碰撞，车道保持则会让车辆行驶速度降低，但是没有碰撞的风险。通过数学建模之后再用实验来标定模型的具体的参数。

    

29. Vehicle Trajectory Prediction using Non-Linear Input- Output Time Series Neural Network

    TV Sushmitha, CP Deepika, R Uppara... - ... Conference on Power ..., 2019 -ieeexplore.ieee.org

    这篇文章感觉有点过于水了，用的是matlab的一个工具箱，Non-Linear Input-Output Neural Networks，其实就是一个两层的神经网络，输入是t-d到t-1时刻的轨迹信息，输出是预测的t时刻轨迹信息。实验用的是CarMaker。

    

30. A game theory-based approach for modelling mandatory lane-changing behaviour in a connected environment

    Y Ali, Z Zheng, MM Haque, M Wang - Transportation research part C ..., 2019 - Elsevier

    这篇文章是采用传统的博弈论对ego vehicle的强制换道行为进行建模，考虑的场景是高速公路汇入，以及传统环境和网联环境。传统环境通过NGSIM的数据对模型进行标定。网联环境是给实验者在驾驶模拟器界面提供了显示车辆速度、邻近车道gap的信息，然后实验者做决策，通过这样的数据对模型参数进行标定。

    这篇论文也提醒我们网联环境下的数据目前基本是没有的，总结的所有数据集都是采集的传统环境下的，所以很难考虑真实网联环境下的车辆换道行为的建模。

    

31. Naturalistic driver intention and path prediction using recurrent neural networks

    A Zyner, S Worrall, E Nebot - IEEE Transactions on Intelligent ..., 2019 - ieeexplore.ieee.org

    这篇论文研究了无标识环状交叉路口的单一车辆轨迹预测，主要架构是LSTM，但是为了输出多个可能的路径，输出层采用的是混合密度网络（Mixture Density Network，基本原理就是输出的是多个高斯概率分布的权重、均值、标准差，从而得到高斯混合模型，采样并聚类后得到车辆的不同轨迹），另外一个贡献在于开放了一个环状交叉路口的数据集，是用车载激光雷达采集的数据，对我们应该用处不大。

    

32. Data-Driven Vehicle Trajectory Forecasting

    S Jawed, E Boumaiza, J Grabocka... - arXiv preprint arXiv ..., 2019 - arxiv.org

    这一篇还是采用深度学习的方法，和之前的论文不同点在于采用的是CNN而不是LSTM，数据采用的是Udacity提供的数据集，包含了车辆前置摄像头拍摄的画面，该论文采用了Faster R-CNN来检测并追踪周围的车辆轨迹信息，然后用CNN处理这些车辆的轨迹信息，CNN输出为预测结果。

    

33. Decision Making and Trajectory Planning of Intelligent Vehicle's Lane-Changing Behavior on Highways under Multi-Objective Constrains

    L Nie, Z Yin, H Huang - 2020 - sae.org

    无法下载

    

34. Prediction performance of lane changing behaviors: a study of combining environmental and eye-tracking data in a driving simulator

    Q Deng, J Wang, K Hillebrand... - IEEE Transactions ..., 2019 - ieeexplore.ieee.org

    这篇文章是预测ego vehicle自身的intent，并设计用户界面。实现了多种算法来进行预测，包括HMM（隐马尔可夫模型）、SVM（支持向量机）、CNN（卷积神经网络）、RF（随机森林）。结果是RF预测效果最好。

    这篇实际上是传统的机器学习方法，将预测问题转换为classification问题来解决，采用的特征是数据采集软件选取的（包括环境数据和眼部追踪数据）。

    

35. An Integrated Approach to Probabilistic Vehicle Trajectory Prediction via Driver Characteristic and Intention Estimation

    J Liu, Y Luo, H Xiong, T Wang... - 2019 IEEE Intelligent ..., 2019 - ieeexplore.ieee.org

    这一篇还是预测单车的换道意图（左转、右转、保持）、司机的驾驶风格、以及驾驶轨迹。

# 性能指标

常用的性能指标为Root Mean Squared Error（RMSE），即均方误差。

另外还有不少用的是Average displacement error（ADE），即预测出的轨迹和真实值的平均差距，有时候也会比较Final displacement error（FDE），即最终预测的位置和真实值的平均差距。各自定义如下：

1. Average displacement error (ADE): The mean Euclidean distance over all the predicted positions and ground truth positions during the prediction time.

2. Final displacement error (FDE): The mean Euclidean distance between the final predicted positions and the corresponding ground truth locations.

# 数据集

## The **High**way **D**rone Dataset

链接：https://www.highd-dataset.com/

介绍：这个是德国亚琛工业大学汽车工程研究所2018年发布的HighD数据集，用于满足自动驾驶基于场景的验证。 他们根据数量、种类和所包含的情景来对数据集进行了评估。数据集包括来自六个地点的11.5小时测量值和110 000车辆，所测量的车辆总行驶里程为45 000 km，还包括了5600条完整的变道记录。

![image]({{site.baseurl}}/assets/2020-05-10-survey/image-20200424140823348.png)

## NGSIM

链接：https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

介绍：Next Generation SIMulation （NGSIM）是美国交通局提供的数据集，广泛用于交通流量预测，驾驶员模型定参，以及车辆轨迹预测等方面的研究。

### US Highway 101 Dataset

其中US Highway 101 Dataset常用于车辆轨迹预测，该数据集提供了2005年美国101号高速公路的车辆轨迹信息。完整的数据集中总共有45分钟的数据，分为三个15分钟的时段：上午7:50至上午8:05；上午8:05至8:20，上午8:20到上午8:35，这些时段代表拥堵的加剧，或者是非拥塞和拥挤状况之间的过渡，以及高峰时段的完全拥塞。除车辆轨迹数据外，US 101数据集还包含计算机辅助设计和地理信息系统文件，航拍矫正的照片，环路探测器数据，原始和处理后的视频，天气数据以及汇总数据分析报告。

![image]({{site.baseurl}}/assets/2020-05-10-survey/07030fig1.jpg)

### Interstate 80 Freeway Dataset

这个和Highway 101 Dataset基本一致，只是在I-80高速公路。

![image]({{site.baseurl}}/assets/2020-05-10-survey/image137b.jpg)

## CARLA Dataset

链接：https://github.com/nrhine1/precog_carla_dataset

介绍：这个是[PRECOG](https://sites.google.com/view/precog)论文在CARLA仿真中得到的数据集，包含了60,701个训练数据，7586个验证数据和7567个测试数据，每个数据都具有频率10Hz的的2秒过去时间和2秒未来时间的位置以及车辆的LIDAR数据。