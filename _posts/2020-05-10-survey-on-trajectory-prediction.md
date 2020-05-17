---
title: "Survey on Vehicle Trajectory Prediction"
last_modified_at: 2020-05-17
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
   
   

## 老师推荐（ref1：24篇，ref2：52篇）

### reference 1

1. A neural network model for driver's lane-changing trajectory prediction in urban traffic flow

   C Ding, W Wang, X Wang, M Baumann - Mathematical Problems in ..., 2013 - hindawi.com

   相对早期的工作，采用了2层神经网络BPNN，输入是一个向量，包含一秒10帧的横向的数据，输出是下一秒10帧的预测。训练方法是Levenberg-Marquardt，（这个是经典的优化算法，只用到了一阶雅克比信息，但是对二阶海森也进行了估计），对比了Elman Network（早期的一种RNN），发现BPNN表现要好一些。

   

2. Trajectory prediction of a lane changing vehicle based on driver behavior estimation and classification

   P Liu, A Kurt - 17th international IEEE conference on ..., 2014 - ieeexplore.ieee.org

   这篇文章并不是预测车辆是否变道，而是预测车辆变道之后它的轨迹是怎么样的，behavior estimation其实就是有一个分支结构对司机的换道轨迹做一个分类（危险或者正常），总体采用的还是基于概率的方案，HMM（Hidden Markov Model）。

   

3. Prediction of lane change trajectories through neural network

   RS Tomar, S Verma, GS Tomar - ... International Conference on ..., 2010 - ieeexplore.ieee.org

   这篇结构更简单，采用的是MLP，上一阶段状态作为输入，下一阶段状态作为输出。

   

4. Predicting driver's lane-changing decisions using a neural network model

   J Zheng, K Suzuki, M Fujita - Simulation Modelling Practice and Theory, 2014 - Elsevier

   这篇文章采用五层MLP来预测target vehicle是否换道，输入是自身车道和相邻两个车道的车辆状态（相对速度、相对距离、类型），然后还分析了实验结果，对重卡车对司机换道的影响作了sensitivity analysis。

   

5. Lane-change detection based on vehicle-trajectory prediction

   H Woo, Y Ji, H Kono, Y Tamura... - IEEE Robotics and ..., 2017 - ieeexplore.ieee.org

   这篇文章重点在于预测target vehicle自身的intent，为了预测intent，作者采用了SVM来分类，特征是和中心线距离、横向加速度、以及一个potential feature（用来刻画周围车辆对target vehicle的影响）。另一方面，作者考虑到intent预测和轨迹预测是很相关的，所以通过机器人领域常用的势场法来预测车辆轨迹（这个是轨迹规划的方法，可以take accounts周围的障碍物），结合这两种方式最终给出target vehicle的intent。

   

6. Lane-changes prediction based on adaptive fuzzy neural network

   J Tang, F Liu, W Zhang, R Ke, Y Zou - Expert Systems with Applications, 2018 - Elsevier

   这篇文章是通过FNN（fuzzy neural network）来预测target vehicle的变道决策，特征还是人为选取，这个adaptive是指通过FNN的预测误差来指导FNN网络更新。

   

7. Lane change trajectory prediction by using recorded human driving data

   W Yao, H Zhao, P Bonnifait... - 2013 IEEE Intelligent ..., 2013 - ieeexplore.ieee.org

   这篇文章基本思想比较简单，相对于以往工作用数学模型生成换道轨迹，这篇文章记录了真实车辆换道的轨迹储存在数据库，然后在当前车辆准备换道的时候，从数据库中找到K个最相近的换道轨迹，然后做插值从而生成新的轨迹。

   

8. Lane changing prediction at highway lane drops using support vector machine and artificial neural network classifiers

   Y Dou, F Yan, D Feng - 2016 IEEE International Conference on ..., 2016 - ieeexplore.ieee.org

   内容和标题差不多。。基本就是深度学习之前的一些套路：处理数据，手动选取特征，分类（SVM或者ANN）。选取的数据有和前车以及后车的速度差、加速度差、距离差。然后在非merge场景下有94%的准确性，merge场景下78%。

   

9. Neural network based lane change trajectory prediction in autonomous vehicles

   RS Tomar, S Verma - Transactions on computational science XIII, 2011 - Springer

   和前面的基于NN的工作没有太大差别，只是介绍的详细一些。

   

10. Real time trajectory prediction for collision risk estimation between vehicles

    S Ammoun, F Nashashibi - 2009 IEEE 5th International ..., 2009 - ieeexplore.ieee.org

    这篇文章是reference 2 第41个的prior work，思路差不多，这个侧重点在于通过轨迹预测来估计碰撞风险，轨迹预 测采用的是Kalman filter.

    

11. Multi-parameter prediction of drivers' lane-changing behaviour with neural network model

    J Peng, Y Guo, R Fu, W Yuan, C Wang - Applied ergonomics, 2015 - Elsevier

    这篇文章主要是做了实验，在实际的车辆上放上传感器，收集司机和车辆的各种信息，然后通过简单的神经网络进行判断到底是lane-keeping还是lane-changing。

    

12. An analysis of the lane changing manoeuvre on roads: the contribution of inter-vehicle cooperation via communication

    S Ammoun, F Nashashibi... - 2007 IEEE Intelligent ..., 2007 - ieeexplore.ieee.org

    重复

    

13. Convolutional social pooling for vehicle trajectory prediction

    N Deo, MM Trivedi - ... of the IEEE Conference on Computer ..., 2018 - openaccess.thecvf.com

    这篇文章目的是预测target vehicle周围车辆的未来轨迹以及类型，总体结构用的是LSTM encoder decoder，创新点在于结构中的Convolutional Social Pooling层，目的是解决普通LSTM中如果直接将ego vehicle周围的车辆信息输入，会有空间位置丢失的问题。

    

14. Game theoretic approach for predictive lane-changing and car-following control

    M Wang, SP Hoogendoorn, W Daamen... - ... Research Part C ..., 2015 - Elsevier

    这篇文章偏向于target vehicle本身的控制，对其他车辆的预测是基于模型的方法，控制策略本身是作为优化问题来解决的，解得的策略是车辆换道的离散序列（即换到哪个道），具体的换道轨迹是三角函数样式的。

    

15. Multivariate time series prediction of lane changing behavior using deep neural network

    J Gao, YL Murphey, H Zhu - Applied Intelligence, 2018 - Springer

    这篇文章提出的模型是一个Group-wise CNN结构，输入是采集的脑电信号、肌电信号等司机的信息，想要从这些信息中判断司机的意图，这篇的创新点在于这个Group-wise，作者认为CNN直接用于时变信号分析缺点是相邻的信号可能没有什么关系，不同于图像信息在空间上是有关系的，所以作者提出对这些信号进行分组，并给出了一种学习分组的算法，从而让相关的信号在一个卷积分支。

    

16. Vehicle trajectory prediction based on motion model and maneuver recognition

    A Houenou, P Bonnifait, V Cherfaoui... - 2013 IEEE/RSJ ..., 2013 - ieeexplore.ieee.org

    这篇文章提出的系统框架还是基于模型和车辆maneuver的，两个主要部分，一个是用CYRA（Constant Yaw and Acceleration)模型直接预测车辆轨迹，另一个是是先识别车辆的maneuver再去根据这个生成车辆轨迹，最后将两种方法生成的轨迹结合得到最终的轨迹。

    

17. Multi-modal trajectory prediction of surrounding vehicles with maneuver based lstms

    N Deo, MM Trivedi - 2018 IEEE Intelligent Vehicles Symposium ..., 2018 - ieeexplore.ieee.org

    重复

    

18. An LSTM network for highway trajectory prediction

    F Altché, A de La Fortelle - 2017 IEEE 20th International ..., 2017 - ieeexplore.ieee.org

    这篇文章采用LSTM预测target vehicle车辆的轨迹，输入包括了周围9辆车的轨迹（每个车道3辆），架构也比较简单，就是一层LSTM网络加上两层fully connected网络。实验给出的预测10s的lateral RMS小于0.7m，纵向速度RMS小于3m/s。

    这一篇引用量比较多，可能在当时说明了基于数据的方法要好于基于模型的方案，但是感觉从现在的角度来看参考价值不大。

    

19. Modeling mandatory lane changing using Bayes classifier and decision trees

    Y Hou, P Edara, C Sun - IEEE Transactions on Intelligent ..., 2013 - ieeexplore.ieee.org

    这个用的是传统的机器学习方法，采用贝叶斯分类器以及决策树，还是选取特征然后分类那一套，实验发现两者结合效果最好（其实就是boosting的概念）。比较有意思的是最后的结果：在非汇合场景中有94.3%的几率正确识别到换道，汇合场景中只有79.3%。之前的几篇采用传统机器学习方法的得到的也差不多是这个结果，可能也说明了在interaction比较强的环境下，这些方法基本是不适用的；当然，这个结果是数据不均衡造成的，毕竟论文中非汇合场景的训练用例有459个，汇合的只有208个，相差一倍多。

    

20. Freeway traffic oscillations: observations and predictions

    M Mauch, MJ Cassidy - ... of the 15th international symposium on ..., 2002 - emerald.com

    这篇文章主要是调查交通流的oscillation受哪些因素的影响，并对其进行了建模，然后给出的结论是在moderately dense的交通流下，oscillation大部分是由于换道行为引起的。这个文章应该可以作为研究换道预测的动机来引用。

    

21. Situation assessment and decision making for lane change assistance using ensemble learning methods

    Y Hou, P Edara, C Sun - Expert Systems with Applications, 2015 - Elsevier

    这个和19是一个作者，这回方法换成了ensemble learning，随机森林和Adaboost，然后结果有了一点提升。

    

22. Recent developments and research needs in modeling lane changing

    Z Zheng - Transportation research part B: methodological, 2014 - Elsevier

    这篇文章是换道模型的研究的一个综述，将这些研究分成了LCD（Lane Change Decision making）和LCI （Lane Change Impact），即换道的行为决策以及换道对周围车辆的影响。最后作者认为我们目前的建模还是不完整的，需要建立一个comprehensive的模型来同时考虑LCD以及LDI。

    

23. Decentralized cooperative lane-changing decision-making for connected autonomous vehicles

    J Nie, J Zhang, W Ding, X Wan, X Chen, B Ran - IEEE Access, 2016 - ieeexplore.ieee.org

    重复

    

24. Stochastic modeling and real-time prediction of vehicular lane-changing behavior

    JB Sheu, SG Ritchie - Transportation Research Part B: Methodological, 2001 - Elsevier
    
    这篇文章是在交通流的角度研究换道，主要是预测在出现交通事故的时候换道情况有什么变化，和我们的研究关系应该不大。

### reference 2

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

    这一篇还是预测单车的，预测单车的换道意图（左转、右转、保持）、司机的驾驶风格、以及驾驶轨迹。基于的方案是DBN（Dynamic Bayesian Network) 以及高斯过程。
    
    
    
36. Integrated deep learning and stochastic car-following model for traffic dynamics on multi-lane freeways

    S Lee, D Ngoduy, M Keyvan-Ekbatani - Transportation research part C ..., 2019 - Elsevier

    这篇文章提出了一个车辆横向运动的模型，采用的是CNN，给出左右换道以及车道保持不变的决策。

    

37. Prediction of Surrounding Vehicles Lane Change Intention Using Machine Learning

    A Benterki, M Boukhnifer, V Judalet... - 2019 10th IEEE ..., 2019 - ieeexplore.ieee.org

    这一篇是传统的机器学习方案来预测车辆的换道意图，分成两步，首先是特征提取，选取的特征是偏航角度以及和车辆和左右车道的距离，然后是用机器学习模型训练和分类，选取了神经网络和支持向量机，总的来说没什么新意。

    

38. Research on Lane-Change Online-Planning for Automotive Vehicles based on Model Predictive Control

    S Li, L Xiao, S Cheng, C Huang... - ... WRC Symposium on ..., 2019 - ieeexplore.ieee.org

    这个是控制方面的文章，考虑的是自动驾驶车辆换道轨迹的生成和跟踪，重点在于构建的框架中轨迹生成和跟踪这两个阶段是不断迭代的，使得模型对环境变化更有适应性。不过和我们想要研究的换道决策及轨迹预测关系应该不大。

    

39. A Personalized Model for Driver Lane-Changing Behavior Prediction Using Deep Neural Network

    J Gao, H Zhu, YL Murphey - 2019 2nd International Conference ..., 2019 - ieeexplore.ieee.org

    这一篇还是预测换道意图，当做分类问题解决的，模型输入是GPS信息、车辆速度、朝向还有司机的生理信息（用的是自己采集的数据），然后模型的架构用的是Root-ResNet（一种CNN）。

    

40. Trajectory Planning and Safety Assessment of Autonomous Vehicles Based on Motion Prediction and Model Predictive Control

    Y Wang, Z Liu, Z Zuo, Z Li, L Wang... - IEEE Transactions on ..., 2019 - ieeexplore.ieee.org

    这篇文章的重点也在于控制，我重点看的是他这里对其他车辆的预测，但是这一部分还是过于简化，假设了其他车辆不会变道，然后用车辆运动模型和蒙特卡洛模拟来获得对其他车辆运动情况的预测结果。

    

41. An analysis of the lane changing manoeuvre on roads: the contribution of inter-vehicle cooperation via communication

    S Ammoun, F Nashashibi... - 2007 IEEE Intelligent ..., 2007 - ieeexplore.ieee.org

    这篇文章中主要把通信作为获取其他车辆信息的手段，当做背景介绍了一下，对于车辆换道模型的建模采用的是5阶多项式来拟合换道轨迹。

    

42. Modeling lane-changing behavior in a connected environment: A game theory approach

    A Talebpour, HS Mahmassani, SH Hamdar - Transportation Research Part ..., 2015 - Elsevier

    这篇文章采用博弈论对ego vehicle的换道决策进行建模，主要考虑了无通信和有通信情况下的ego vehicle和lag vehicle的零和博弈，这里假设通信给出的信息包含ego vehicle周围车辆的换道决策，而这恰恰是我们想要解决的问题。。

    

43. Design and efficiency measurement of cooperative driver assistance system based on wireless communication devices

    S Ammoun, F Nashashibi - Transportation research part C: emerging ..., 2010 - Elsevier

    这篇文章采用802.11p协议（WAVE）来研究车辆间通信对ITS的影响，选取了两个场景，一个是交叉路口的碰撞预警，另一个是车辆换道的决策辅助。两个场景用的信息均为通过车辆通信获取的其他车辆GPS信号，在换道决策辅助场景中，采用了5阶多项式来拟合车辆换道轨迹，并增加了诸如横/纵向加速度限制等约束，从而给出3个层次的换道预警：

    红色：如果车主现在进行换道，那么很可能发生碰撞

    橙色：如果车主现在换道且其他车辆行驶轨迹不变，那么5s内可能发生碰撞

    绿色：如果车主现在换道，10s内应该不会发生碰撞

    这篇文章讨论的是本车的换道预警，对其他车辆的轨迹预测是比较基本的，也没有考虑到车辆之间的互动性对车辆轨迹的影响，因此如果在真实道路上，换道预警可能会有很多false alarm，导致车主选择忽视这个信息。

    

44. Decentralized cooperative lane-changing decision-making for connected autonomous vehicles

    J Nie, J Zhang, W Ding, X Wan, X Chen, B Ran - IEEE Access, 2016 - ieeexplore.ieee.org

    这篇文章考虑的是在fully connected and automated 环境下，每辆CAV如何进行换道决策，这里的假设是通信无时延、换道行为是瞬时的。然后提出的框架包括3个部分：相关车辆状态预测、候选决策生成、以及和其他车辆协同。状态预测是采用基于以往文献中提出的基于模型的方法，类似于IDM，不过考虑了target vehicle前方的多个车辆状态。

    

45. Investigation of cooperative driving behaviour during lane change in a multi-driver simulation environment

    M Heesen, M Baumann, J Kelsch... - Human Factors and ..., 2012 - researchgate.net

    这篇文章比较有意思，作者招募了平均驾龄为13年的司机来做实验考察换道决策和协同换道行为受到哪些因素的影响，实验环境是驾驶模拟器，每次实验包含了2个人类参与者，一个决定是否换道，另一个决定是否协同换道。实验发现对于进行换道决策的司机而言，他们会考虑到协同换道的司机存在的选择。对于协同换道的司机而言，能否预测/解析想要换道的司机的意图是影响他们是否进行协同换道的关键因素。作者给出的建议是如果能够给出其他车辆的换道概率，有助于司机进行更好的协同换道决策（也说明了我们希望进行的研究是比较有意义的）。

    

46. Tracking and behavior reasoning of moving vehicles based on roadway geometry constraints

    K Jo, M Lee, J Kim, M Sunwoo - IEEE Transactions on ..., 2016 - ieeexplore.ieee.org

    这一篇和我们的目的是相似的，主要是通过跟踪周围车辆的轨迹来预测他们的intention，跟踪采用的是车辆自身传感器给出的数据，然后结合road geometry信息给出了curvilinear坐标系（曲线坐标系）下车辆的坐标，intention的预测采用的是基于模型的方法，采用了多个模型（CVLK，CALK，CVLC，CALC）以及IMM（interacting multiple model）来预测。

    这里面实验是实车验证，跟踪的真值是target vehicle的高精度GPS信息，其实在车联网的环境下，其他车辆的GPS信息是可以直接通过通信拿到的，在车辆均有OBU的情况下应该会好于这篇文章的方法。

    

47. Minimizing the disruption of traffic flow of automated vehicles during lane changes

    D Desiraju, T Chantem... - IEEE Transactions on ..., 2014 - ieeexplore.ieee.org

    这篇文章考虑的背景是fully connected and automated，他想要最大化所有车辆换道的次数从而增加交通吞吐量并且减少交通拥堵，他这个主要考虑的还是换道决策而不是预测，和我们的目标关系不大。

    

48. Carrot and stick: A game-theoretic approach to motivate cooperative driving through social interaction

    M Zimmermann, D Schopf, N Lütteken, Z Liu... - ... Research Part C ..., 2018 - Elsevier

    这篇文章采用驾驶模拟器研究如何设置游戏策略来让驾驶员更愿意进行协同驾驶，更偏向心理学，不涉及对车辆轨迹的预测。

    

49. Cooperative driving and lane changing modeling for connected vehicles in the vicinity of traffic signals: A cyber-physical perspective

    Y He, D Sun, M Zhao, S Cheng - IEEE Access, 2018 - ieeexplore.ieee.org

    这篇文章主要是结合智能驾驶员模型和v2x通信下的换道决策（在换道不安全的时候发送换道换道请求给周围的车辆）来建模协同驾驶。

    

50. Acquisition of Relative Trajectories of Surrounding Vehicles using GPS and SRC based V2V Communication with Lane Level Resolution.

    Z Peng, S Hussain, MI Hayee, M Donath - VEHITS, 2017 - scitepress.org

    这篇文章的目的是靠标准的GPS和车辆间通信获取ego vehicle周围车辆的车道级别轨迹，基本思想是由于车辆的GPS误差主要来源于大气效应，而相邻的车辆这种误差是相近的，所以尽管绝对位置的误差较大，通过通信获取其他车辆的GPS后减去本车的可以使得这个误差被消除，从而获取相邻车辆更为精确的车道级别轨迹。

    

51. Model predictive control–based cooperative lane change strategy for improving traffic flow

    D Wang, M Hu, Y Wang, J Wang... - Advances in ..., 2016 - journals.sagepub.com

    这篇文章是中心化控制，采用V2V通信和MPC，通过构建一个优化问题来实现协同换道行为，target vehicle的换道主要考虑邻近车辆可接受的加速度和减速度。

    

52. Cooperative lane change model for connected vehicles under congested traffic conditions

    D Wang, J Wang, Y Wang, S Tian - CICTP 2015, 2015 - ascelibrary.org

    无法下载

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