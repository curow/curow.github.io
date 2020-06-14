---
title: "用Pytorch实现轨迹预测整体框架"
last_modified_at: 2020-06-14
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

本周的主要工作是学习Pytorch这一深度学习框架（之前用的是tensorflow，但是Pytorch目前在学术研究中更为常用，因此花了一定的时间去熟悉Pytorch），然后将Argoverse的轨迹数据集按照Pytorch的规范进行加载（自定义了一个数据类，继承了torch.util.data.Dataset），最后实现了加载数据、模型定义、训练与验证这一系列步骤，下面分别进行阐述。

## 数据处理

之前提过Argoverse的轨迹数据集均为CSV文件，然后每个文件都记录了5s的目标车辆以及其周围车辆的轨迹，每行的信息包含了6类：TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME，这里我们暂时只使用OBJECT_TYPE为AGENT的车辆其X和Y的轨迹，即只用他自身的2s轨迹来预测其接下来3s的轨迹。

下面是我们的数据集实现（详细实现见[项目源代码](https://github.com/curow/vehicle-trajectory-prediction/blob/master/data.py)），这里继承了Pytorch的Dataset类，从而让接下来的数据加载和模型训练更为方便：

```python
class TrajectoryDataset(Dataset):
    
    def __init__(self, root_dir, mode):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.sequences = [(self.root_dir / x).absolute() for x in os.listdir(self.root_dir)]
        self.obs_len = 20
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = pd.read_csv(self.sequences[idx])
        agent_x = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["X"]
        agent_y = sequence[sequence["OBJECT_TYPE"] == "AGENT"]["Y"]
        agent_traj = np.column_stack((agent_x, agent_y)).astype(np.float32)
        # return input and target
        return [agent_traj[:self.obs_len], agent_traj[self.obs_len:]]

```

## 加载数据

Pytorch中有DataLoader类，有了上面的数据集实现，我们就可以直接用DataLoader来每次获取指定数量的数据，代码如下：

```python
BATCH_SIZE = 16
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=6)
```

## 模型定义

这里我们用最简单的多层感知机模型，因为本周的重点是调通整个流程，下周再编写更为强大和更适合序列数据的模型（如LSTM编解码器，Transformer等），代码如下：

```python
class MLP(nn.Module):
    """Expected input is (batch_size, 20, 2)
    20: input sequence length
    2: the dimension of input feature (x and y)
    output shape: (batch_size, 30 * 2)
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20 * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 30 * 2)
        )
    
    def forward(self, x):
        # convert (batch_size, 20, 2) to (batch_size, 20 * 2)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
```

## 训练与验证

上述准备步骤完成之后我们就可以进行模型的训练了，基本流程是每次从DataLoader中获取batch size的数据，然后将输入传入模型，并将输出和目标通过损失函数来进行对比，然后进行误差的反向传播，最后用经典的Adam算法来更新模型权重。为了加快训练速度，我们尽量将模型和数据都放到GPU上进行训练。代码如下：

```python
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
model.to(dev)

epoches = 20 
print("start training...")
for epoch in range(epoches):
    start = time.time()
    model.train()
    for i, (xb, yb) in enumerate(train_loader):
        xb = xb.to(dev)
        yb = yb.to(dev).view(yb.size(0), -1)
        
        optimizer.zero_grad()
        yb_pred = model(xb)
        loss = loss_fn(yb_pred, yb)
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print("epoch {}, round {}/{} train loss: {:.4f}".format(epoch, i, len(train_loader), loss.item()))
            
    model.eval()
    model_dir = "saved_model/MLP"
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), model_dir + "/MLP_epoch{}".format(epoch))
    print("start validating...")
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb.to(dev)), yb.to(dev).view(yb.size(0), -1)) for xb, yb in val_loader)
    print("epoch {}, val loss: {:.4f}, time spend: {}s".format(
            epoch, val_loss / len(val_loader), time.time() - start))

```

## 总结

本周采用Pytorch实现了整套深度学习模型训练的流程，本身的过程不是很复杂，但是涉及到相当多的概念和细节。该流程实现之后，下周的主要工作就是编写更为有效的模型来利用已有的轨迹对其未来轨迹预测进行预测，另一个方面的工作就是将更多的信息作为模型的输入（如其他车辆的轨迹，地图信息等），如何将这些信息整合到模型中也需要多思考并参考最新的相关文献。