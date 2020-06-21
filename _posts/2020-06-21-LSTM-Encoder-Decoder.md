---
title: "用Pytorch实现Encoder Decoder模型"
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

本周主要实现了经典的Encoder Decoder模型，并进一步优化了训练和测试相关代码。

## Encoder Decoder简介

LSTM Encoder Decoder最早由这篇2014年的经典paper提出：[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)，现在的引用量已经过万了。目前该模型也是最常见的sequence-to-sequence模型，基本思想是用一个RNN网络（编码器）来将输入的序列编码为一个固定长度的向量（context vector），该向量可以被视为整个输入序列的一个抽象表示，然后将该向量作为另一个RNN网络（解码器）的初始输入，并输出任意长度的目标序列。

在我们的轨迹预测任务中，输入就是已经观测到的车辆轨迹信息，输出是预测的未来该车辆轨迹。

下面分别介绍Encoder和Decoder网络的编写。

## Encoder

Encoder采用了一层全连接层，四层LSTM，并且采用了dropout来降低过拟合（和原论文保持一致）。可以看到Encoder的编写还是较为简单的，由于我们的输入是3维的tensor，形状为[序列长度，批长度，特征长度]，pytorch的LSTM网络会自动循环读入输入序列，并给出每次循环的网络输出以及最后一次网络的hidden state以及cell state。

```python
class Encoder(nn.Module):
    def __init__(self,
                 input_size = 2,
                 embedding_size = 128,
                 hidden_size = 256,
                 n_layers = 4,
                 dropout = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                           dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: input batch data, size: [sequence len, batch size, feature size]
        for the argoverse trajectory data, size(x) is [20, batch size, 2]
        """
        # embedded: [sequence len, batch size, embedding size]
        embedded = self.dropout(F.relu(self.linear(x)))
        # you can checkout https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM
        # for details of the return tensor
        # briefly speaking, output coontains the output of last layer for each time step
        # hidden and cell contains the last time step hidden and cell state of each layer
        # we only use hidden and cell as context to feed into decoder
        output, (hidden, cell) = self.rnn(embedded)
        # hidden = [n layers * n directions, batch size, hidden size]
        # cell = [n layers * n directions, batch size, hidden size]
        # the n direction is 1 since we are not using bidirectional RNNs
        return hidden, cell
```

## Decoder

Decoder的结构和Encoder的结构也是基本一致的，区别在于，Decoder每次接受输入的序列长度只有1，然后每次Decoder的输出都作为我们下一个时间点的预测。

```python
class Decoder(nn.Module):
    def __init__(self,
                 output_size = 2,
                 embedding_size = 128,
                 hidden_size = 256,
                 n_layers = 4,
                 dropout = 0.5):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        x : input batch data, size(x): [batch size, feature size]
        notice x only has two dimensions since the input is batchs
        of last coordinate of observed trajectory
        so the sequence length has been removed.
        """
        # add sequence dimension to x, to allow use of nn.LSTM
        # after this, size(x) will be [1, batch size, feature size]
        x = x.unsqueeze(0)

        # embedded = [1, batch size, embedding size]
        embedded = self.dropout(F.relu(self.embedding(x)))

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hidden size]
        #hidden = [n layers, batch size, hidden size]
        #cell = [n layers, batch size, hidden size]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # prediction = [batch size, output size]
        prediction = self.linear(output.squeeze(0))

        return prediction, hidden, cell
```

## Seq2Seq

最终，我们的Seq2Seq的模型需要结合Encoder和Decoder，每一次forward都是之前讲到的流程，Encoder将输入的20个序列编码为一个context vector，然后将其作为Decoder的初始输入，并将Encoder最终的hidden state和cell state作为Decoder初始的hidden state和cell state，最终我们在for循环里每次利用Decoder来预测下一个时间点的预测，最终将所有的预测（30个）输出。

这里我们也采用了常见的训练技巧，teacher forcing，即训练的时候Decoder的输入按照一定概率为上一次的输出或者真实的当前时间点的数据，这样做是为了让网络更容易训练，因为序列数据的预测是基于上次时间点的预测，如果上一次都是错的，那么下次可能错的更厉害，所以这里有时候告诉网络真实的预测数据，从而使其不会错上加错。

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, x, y, teacher_forcing_ratio = 0.5):
        """
        x = [observed sequence len, batch size, feature size]
        y = [target sequence len, batch size, feature size]
        for our argoverse motion forecasting dataset
        observed sequence len is 20, target sequence len is 30
        feature size for now is just 2 (x and y)

        teacher_forcing_ratio is probability of using teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        """
        batch_size = x.shape[1]
        target_len = y.shape[0]
        
        # tensor to store decoder outputs of each time step
        outputs = torch.zeros(y.shape).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)

        # first input to decoder is last coordinates of x
        decoder_input = x[-1, :, :]
        
        for i in range(target_len):
            # run decode for one time step
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            # place predictions in a tensor holding predictions for each time step
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_forcing = random.random() < teacher_forcing_ratio

            # output is the same shape as input, [batch_size, feature size]
            # so we can use output directly as input or use true lable depending on
            # teacher_forcing is true or not
            decoder_input = y[i] if teacher_forcing else output

        return outputs
```

## Train

训练流程和上次的MLP基本一致，只是将流程方面优化了一些，写的更加通用化了。

### Loading Data

这里由于LSTM默认接收的输入是sequence 在前，batch在后，所以需要做一下预处理。

```python
class WrappedDataLoader:
    def __init__(self, dataloader, func):
        self.dataloader = dataloader
        self.func = func
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        iter_dataloader = iter(self.dataloader)
        for batch in iter_dataloader:
            yield self.func(*batch)
            
def preprocess(x, y):
    # x and y is [batch size, seq len, feature size]
    # to make them work with default assumption of LSTM,
    # here we transpose the first and second dimension
    # return size = [seq len, batch size, feature size]
    return x.transpose(0, 1), y.transpose(0, 1)
    
train_data, val_data, test_data = get_dataset(["train", "val", "test"])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=6)

train_loader = WrappedDataLoader(train_loader, preprocess)
val_loader = WrappedDataLoader(val_loader, preprocess)
```

### define model

然后我们设置模型的架构参数

```python
INPUT_DIM = 2
OUTPUT_DIM = 2
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Seq2Seq(enc, dec, dev).to(dev)
```

### train and eval

和之前的主要变换就是将train和eval的流程分离开了，因为SeqSeq模型在train和eval的流程方面还是有一定不同的，在eval的时候我们并不会使用teacher forcing（保证eval的损失是客观的）

```python
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(dataloader):
        # put data into GPU
        x = x.to(dev)
        y = y.to(dev)
        
        # zero all param gradients
        optimizer.zero_grad()
        
        # run seq2seq to get predictions
        y_pred = model(x, y)
        
        # get loss and compute model trainable params gradients though backpropagation
        loss = criterion(y_pred, y)
        loss.backward()
        
        # update model params
        optimizer.step()
        
        # add batch loss, since loss is single item tensor
        # we can get its value by loss.item()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(dev)
            y = y.to(dev)
            
            # turn off teacher forcing
            y_pred = model(x, y, teacher_forcing_ratio = 0)
            
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

N_EPOCHES = 100
best_val_loss = float('inf')

# load previous best model params if exists
model_dir = "saved_models/Seq2Seq"
saved_model_path = model_dir + "/best_seq2seq.pt"
if os.path.isfile(saved_model_path):
    model.load_state_dict(torch.load(saved_model_path))
    print("successfully load previous best model parameters")
    
for epoch in range(N_EPOCHES):
    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    
    end_time = time.time()
    
    mins, secs = epoch_time(start_time, end_time)
    
    print(F'Epoch: {epoch+1:02} | Time: {mins}m {secs}s')
    print(F'\tTrain Loss: {train_loss:.3f}')
    print(F'\t Val. Loss: {val_loss:.3f}')

    if val_loss < best_val_loss:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), saved_model_path)
```

