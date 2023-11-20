import random
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


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class ConvNormPool(nn.Module):
    """Conv Skip-connection module"""
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        norm_type='bachnorm'
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.conv_3 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size
        )
        self.swish_1 = Swish()
        self.swish_2 = Swish()
        self.swish_3 = Swish()
        if norm_type == 'group':
            self.normalization_1 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_2 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
            self.normalization_3 = nn.GroupNorm(
                num_groups=8,
                num_channels=hidden_size
            )
        else:
            self.normalization_1 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_2 = nn.BatchNorm1d(num_features=hidden_size)
            self.normalization_3 = nn.BatchNorm1d(num_features=hidden_size)
            
        self.pool = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, input):
        conv1 = self.conv_1(input)
        x = self.normalization_1(conv1)
        x = self.swish_1(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        x = self.conv_2(x)
        x = self.normalization_2(x)
        x = self.swish_2(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))
        
        conv3 = self.conv_3(x)
        x = self.normalization_3(conv1+conv3)
        x = self.swish_3(x)
        x = F.pad(x, pad=(self.kernel_size - 1, 0))   
        
        x = self.pool(x)
        return x


class CNN(nn.Module):
    def __init__(
        self,
        input_size = 1,
        hid_size = 256,
        kernel_size = 4,
        num_classes = 4,
    ):
        
        super().__init__()
        
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size//2,
            kernel_size=kernel_size,
        )
        self.conv3 = ConvNormPool(
            input_size=hid_size//2,
            hidden_size=hid_size//4,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size//4, out_features=num_classes)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)        
        # print(x.shape) # num_features * num_channels
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)
        return x


class RNN(nn.Module):
    """RNN module(cell type lstm or gru)"""
    def __init__(
        self,
        input_size,
        hid_size,
        num_rnn_layers=1,
        dropout_p = 0.2,
        bidirectional = False,
        rnn_type = 'lstm',
    ):
        super().__init__()
        
        if rnn_type == 'lstm':
            self.rnn_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            
        else:
            self.rnn_layer = nn.GRU(
                input_size=input_size,
                hidden_size=hid_size,
                num_layers=num_rnn_layers,
                dropout=dropout_p if num_rnn_layers>1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
    def forward(self, input):
        outputs, hidden_states = self.rnn_layer(input)
        return outputs, hidden_states
    

class RNNModel(nn.Module):
    def __init__(
        self,
        input_size,
        hid_size,
        rnn_type,
        bidirectional,
        n_classes=4,
        kernel_size=4,
    ):
        super().__init__()
            
        self.rnn_layer = RNN(
            input_size=46,#hid_size * 2 if bidirectional else hid_size,
            hid_size=hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x, _ = self.rnn_layer(x)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=1)#.squeeze(1)
        return x


class RNNAttentionModel(nn.Module):
    def __init__(
        self,
        input_size,
        hid_size,
        rnn_type,
        bidirectional,
        n_classes=4,
        kernel_size=4,
    ):
        super().__init__()
 
        self.rnn_layer = RNN(
            input_size=1112,
            hid_size=hid_size,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        self.conv1 = ConvNormPool(
            input_size=input_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )
        self.conv2 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )

        self.conv3 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )

        self.conv4 = ConvNormPool(
            input_size=hid_size,
            hidden_size=hid_size,
            kernel_size=kernel_size,
        )

        self.dropout = nn.Dropout(p=0.2)

        self.avgpool = nn.AdaptiveMaxPool1d((1))
        self.attn = nn.Linear(hid_size, hid_size, bias=False)
        self.fc = nn.Linear(in_features=hid_size, out_features=n_classes)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)


        x_out, hid_states = self.rnn_layer(x)
        # x_out = self.dropout(x_out)
        
        x = torch.cat([hid_states[-1]], dim=0).transpose(0, 1)
        # x = self.dropout(x)

        x_attn = torch.tanh(self.attn(x))
        x = x_attn.bmm(x_out)
        x = x.transpose(2, 1)
        x = self.avgpool(x)
        x = x.view(-1, x.size(1) * x.size(2))
        x = F.softmax(self.fc(x), dim=-1)
        return x


def get_all_data(X_train, y_train, batch_size):
    y_tensor = torch.tensor([y_train], dtype=torch.int32)
    y_tensor = torch.transpose(y_tensor, 0, 1)
    X_tensor = torch.tensor([X_train], dtype=torch.float32)
    X_tensor = torch.transpose(X_tensor, 0, 1)

    _, counts = np.unique(y_train, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[e] for e in y_train.flatten()]
    sampler = WeightedRandomSampler(example_weights, X_train.shape[0])
    data_loader = DataLoader(MyDataset(X_tensor, y_tensor), batch_size=batch_size) #, sampler=sampler

    return data_loader


def predict(X_test):
    nsamples, n_features = X_test.shape[0], X_test.shape[1]

    model = RNNAttentionModel(1, 64, 'lstm', n_classes=4, kernel_size=4, bidirectional=False).to(device)
    model.load_state_dict(torch.load('models/rnn_model_weights.pth'))
    model.eval()

    X_test = torch.tensor([X_test], dtype=torch.float32)
    X_test = torch.transpose(X_test, 0, 1).to(device)
    predictions = []

    current_batch = 0
    for iteration in range(nsamples):
        batch_x = X_test[current_batch: current_batch + iteration]
        if len(batch_x) > 0:
            output = model(batch_x)
            _, pred = torch.max(output.data, 1)
            preds = pred.cpu().numpy()
            predictions.extend(preds)

        current_batch += iteration

    return predictions


def train(X_train, y_train, X_val, y_val, print_iter=20):
    batch_size = 32
    n_epochs = 40
    lr = 1e-2
    
    data_loader = get_all_data(X_train, y_train, batch_size)
    X_val = torch.tensor([X_val], dtype=torch.float32)
    X_val = torch.transpose(X_val, 0, 1).to(device)

    model = RNNAttentionModel(1, 64, 'lstm', n_classes=4, kernel_size=4, bidirectional=False).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=5e-6)
    
    model.train(True)

    for epoch in range(n_epochs):
        for iteration, (batch_x, batch_y) in enumerate(data_loader):
            a, b = random.randint(0, 17806/2), random.randint(int(17806/3), 17806)
            batch_x = batch_x[:, :, a:(a+b)]
            batch_x = batch_x.to(device)
            # print(batch_x.size())
            batch_y = torch.flatten(batch_y).long().to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            if iteration % print_iter == 0:
                _, pred = torch.max(output.data, 1)
                preds = pred.cpu().numpy()
                y_vals = batch_y.cpu().numpy()
                print('Train Iter / Epoch / Num epochs: {:03d}/{}/{}....'.format(iteration, epoch, n_epochs), end=' ')
                print("Loss: {:.4f} F1 {:.5f}".format(loss.item(), f1_score(y_vals, preds, average='micro')))
                    
        with torch.no_grad():
            model.eval()
            predictions = []
            current_batch = 0
            for iteration in range(X_val.shape[0]):
                batch_x = X_val[current_batch: current_batch + iteration]
                if len(batch_x) > 0:
                    output = model(batch_x)
                    _, pred = torch.max(output.data, 1)
                    preds = pred.cpu().numpy()
                    predictions.extend(preds)

                current_batch += iteration

            print('Eval Epoch / Num epochs: {}/{}....'.format(epoch, n_epochs), end=' ')
            print(" F1 {:.5f}".format(f1_score(y_val, predictions, average='micro')))
            
            model.train(True)
                    
        # scheduler.step()
                

    torch.save(model.state_dict(), 'models/rnn_model_weights.pth')

