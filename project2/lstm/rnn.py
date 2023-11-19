import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import  DataLoader, WeightedRandomSampler, Dataset


device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


class MyDataset(Dataset):

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class CNNBlock(nn.Module):
    def __init__(self, input_channels: int = 1, internal_channels: int = 12, stride: int = 1, kernel_size: int = 3):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, internal_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(internal_channels, internal_channels, kernel_size, stride)
        self.p1 = nn.MaxPool1d(4, 2, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        c1_r = F.relu(c1)

        c2 = self.conv2(c1_r)
        c2_r = F.relu(c2)
        return self.p1(c2_r)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.hidden_size = 64
        self.n_layers = 1
        self.attention_size = 32

        self.cnn1 = CNNBlock(1, 12)
        self.cnn2 = CNNBlock(12, 24)
        self.cnn3 = CNNBlock(24, 36)
        # self.cnn4 = CNNBlock(36, 48)
        # self.cnn5 = CNNBlock(48, 60)
        # self.cnn6 = CNNBlock(60, 72)
        self.rnn = nn.LSTM(36, self.hidden_size, self.n_layers, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.attention_size)
        self.bn = nn.BatchNorm1d(self.attention_size)
        self.fc = nn.Linear(self.attention_size, 4)

    def forward(self, x):
        batch_size = x.size(0)

        c1 = self.cnn1(x)
        c2 = self.cnn2(c1)
        c3 = self.cnn3(c2)
        # c4 = self.cnn4(c3)
        # c5 = self.cnn5(c4)
        # c6 = self.cnn6(c5)

        c3_t = torch.transpose(c3, 1, 2)
        c3_t = F.dropout(c3_t, 0.3)

        hidden_h = torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float32).to(device)
        hidden_c = torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float32).to(device)

        out_all, (hidden, hidden_c) = self.rnn(c3_t, (hidden_h, hidden_c))

        attention = torch.zeros(batch_size, self.attention_size).to(device)
        for i in range(len(out_all[0])):
            attention = attention + self.attn(out_all[:, i])

        bn_attention = self.bn(attention)

        out = self.fc(bn_attention)

        return out


def get_all_data(X_train, y_train, batch_size):
    y_tensor = torch.tensor([y_train], dtype=torch.int32)
    y_tensor = torch.transpose(y_tensor, 0, 1)
    X_tensor = torch.tensor([X_train], dtype=torch.float32)
    X_tensor = torch.transpose(X_tensor, 0, 1)

    _, counts = np.unique(y_train, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[e] for e in y_train.flatten()]
    sampler = WeightedRandomSampler(example_weights, X_train.shape[0])
    data_loader = DataLoader(MyDataset(X_tensor, y_tensor), batch_size=batch_size,
                             sampler=sampler
    )

    return data_loader


def predict(X_test, n_epochs: int = 20,
              n_hidden: int = 128, lr: float = 0.01,
              batch_size=56, print_iter=2, n_layers=1, output_size=4):
    nsamples, n_features = X_test.shape[0], X_test.shape[1]

    rnn = RNN()
    rnn.load_state_dict(torch.load('models/rnn_model_weights.pth'))
    rnn.eval()

    X_test = torch.tensor([X_test], dtype=torch.float32)
    X_test = torch.transpose(X_test, 0, 1)
    predictions = []

    current_batch = 0
    for iteration in range(nsamples):
        batch_x = X_test[current_batch: current_batch + iteration]
        if len(batch_x) > 0:
            output = rnn(batch_x)
            _, pred = torch.max(output.data, 1)
            preds = pred.cpu().numpy()
            predictions.extend(preds)

        current_batch += iteration

    return predictions


def train_rnn(X_train, y_train, n_epochs: int = 20, print_iter=20):
    batch_size = 128
    data_loader = get_all_data(X_train, y_train, batch_size)

    model = RNN().to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=5e-6)
    
    model.train(True)

    for epoch in range(n_epochs):
        for iteration, (batch_x, batch_y) in enumerate(data_loader):
            batch_x = batch_x.to(device)
            batch_y = torch.flatten(batch_y).long().to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            if iteration % print_iter == 0:
                with torch.no_grad():
                    model.eval()
                   
                    _, pred = torch.max(output.data, 1)
                    preds = pred.cpu().numpy()
                    y_val = batch_y.cpu().numpy()
                    print('Iter / Epoch / Num epochs: {:03d}/{}/{}....'.format(iteration, epoch, n_epochs), end=' ')
                    print("Loss: {:.4f} F1 {:.5f}".format(loss.item(), f1_score(y_val, preds, average='micro')))
                    
                    model.train(True)
                    
        scheduler.step()
                

    torch.save(model.state_dict(), 'models/rnn_model_weights.pth')
