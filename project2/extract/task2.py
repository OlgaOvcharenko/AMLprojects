############# imports
import random
import numpy as np
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import  DataLoader, WeightedRandomSampler, Dataset
import copy


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsample = downsample




    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            out = self.maxpool(out)
            identity = self.downsample(x)

        return out




class ECGNet(nn.Module):

    def __init__(self, struct=[15, 17, 19, 21], in_channels=8, fixed_kernel_size=17, num_classes=34):
        super(ECGNet, self).__init__()
        self.struct = struct
        self.planes = 16
        self.parallel_conv = nn.ModuleList()

        for i, kernel_size in enumerate(struct):
            sep_conv = nn.Conv1d(in_channels=in_channels, out_channels=self.planes, kernel_size=kernel_size,
                               stride=1, padding=0, bias=False)
            self.parallel_conv.append(sep_conv)
        self.parallel_conv.append(nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(in_channels=1, out_channels=self.planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        ))

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=fixed_kernel_size,
                               stride=2, padding=2, bias=False)
        self.block = self._make_layer(kernel_size=fixed_kernel_size, stride=1, padding=8)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=8, stride=8, padding=2)
        self.rnn = nn.LSTM(input_size=8, hidden_size=40, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(in_features=680, out_features=num_classes)


    def _make_layer(self, kernel_size, stride, blocks=15, padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            if (i + 1) % 4 == 0:
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=1,
                               stride=1, padding=0, bias=False),
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes + base_width, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
                self.planes += base_width
            elif (i + 1) % 2 == 0:
                downsample = nn.Sequential(
                    nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                )
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))
            else:
                downsample = None
                layers.append(ResBlock(in_channels=self.planes, out_channels=self.planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, downsample=downsample))

        return nn.Sequential(*layers)



    def forward(self, x):
        out_sep = []

        for i in range(len(self.struct)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  # out => [b, 16, 9960]

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  # out => [b, 64, 10]
        out = out.reshape(out.shape[0], -1)  # out => [b, 640]

        rnn_out, (rnn_h, rnn_c) = self.rnn(x.permute(2, 0, 1))
        new_rnn_h = rnn_h[-1, :, :]  # rnn_h => [b, 40]

        new_out = torch.cat([out, new_rnn_h], dim=1)  # out => [b, 680]
        result = self.fc(new_out)  # out => [b, 20]

        # print(out.shape)

        return result


class SE_Module(nn.Module):

    def __init__(self, in_channels, ratio=16, dim=2):
        super(SE_Module, self).__init__()
        self.dim = dim
        if self.dim == 1:
            self.squeeze = nn.AdaptiveAvgPool1d(1)
        else:
            self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // ratio),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=in_channels // ratio, out_features=in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x => [b, c, h, w] / x => [b, c, l]
        identity = x

        out = self.squeeze(x)
        out = out.reshape(out.shape[0], out.shape[1])
        scale = self.excitation(out)
        if self.dim == 1:
            scale = scale.reshape(scale.shape[0], scale.shape[1], 1)
        else:
            scale = scale.reshape(scale.shape[0], scale.shape[1], 1, 1)

        return identity * scale.expand_as(identity)



class ResBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, downsample=None,downsample2=None, num_conv=1):
        super(ResBlock1d, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False,dtype=torch.float)
        self.SE1 = SE_Module(in_channels=out_channels,dim=1)
        self.num_conv = num_conv
        self.dropout = nn.Dropout(.2)
        if self.num_conv == 3:
            self.bn2 = nn.BatchNorm1d(num_features=out_channels)
            self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=False,dtype=torch.float)
            self.bn3 = nn.BatchNorm1d(num_features=out_channels)
            self.conv3 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=False,dtype=torch.float)


    def forward(self, x):
        identity=x
        out = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(out)
        print(out.shape)

        return out
        

class SE_ECGNet(ECGNet):

    def __init__(self, struct=[ 300,  500,  700], num_classes=4):
        super(ECGNet, self).__init__()
        self.struct = struct
        
        ############ let's replicate the medium article############
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6,stride=1, padding=0,
                              bias=False , dtype=torch.float)
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6,stride=1, padding=0,
                              bias=False , dtype=torch.float)
        self.conv_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6,stride=1, padding=0,
                              bias=False , dtype=torch.float)
        
        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        
        self.relu_1 = nn.ReLU(inplace=False)
        self.relu_2 = nn.ReLU(inplace=False)
        self.relu_3 = nn.ReLU(inplace=False)
        
        self.avgPool_1= nn.AvgPool1d(kernel_size=3,stride=2)
        
        self.avgPool_2= nn.AvgPool1d(kernel_size=2,stride=2)
        
        self.avgPool_3= nn.AvgPool1d(kernel_size=2,stride=2)
        self.lin_1= nn.Linear(in_features=64*23,out_features=64, dtype=torch.float)
        
        self.lin_2= nn.Linear(in_features=64,out_features=32, dtype=torch.float)
        
        self.lin_3= nn.Linear(in_features=32,out_features=num_classes, dtype=torch.float)
        self.relu_4 = nn.ReLU(inplace=False)
        
        self.relu_5 = nn.ReLU(inplace=False)
        self.softmax= nn.Softmax(dim=1)
        
        
        #edited originally in::1, out 32
        self.conv = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=20, stride=1, padding=0,
                              bias=False , dtype=torch.float)
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm1d(256)
        
        self.relu2 = nn.ReLU(inplace=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=15, stride=1, padding=0,
                              bias=False , dtype=torch.float)
        self.relu3 = nn.ReLU(inplace=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=0,
                              bias=False , dtype=torch.float)
        self.relu4 = nn.ReLU(inplace=False)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0,
                              bias=False , dtype=torch.float)
        self.fc_1 = nn.Linear(in_features=64, out_features=4,dtype=torch.float)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=256 * len(struct), out_features=4,dtype=torch.float)
        self.block1 = self._make_layer(in_channels=256, out_channels=256, kernel_size=15, stride=1,
                                       block=ResBlock1d, blocks=3, padding=0)

        self.block2_list = nn.ModuleList()
        self.block3_list = nn.ModuleList()

        for i, kernel_size in enumerate(self.struct):
            block2 = self._make_layer(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=1,
                                      block=ResBlock1d, blocks=4, padding=0)
            block3 = self._make_layer(in_channels=256, out_channels=256, kernel_size=kernel_size, stride=2,
                                      block=ResBlock1d, blocks=4, padding=1 )
            self.block2_list.append(block2)
            self.block3_list.append(block3)

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, block, blocks, padding=0 ):
        layers = []
        num_conv = 1
        downsample = None
        downsample2 = None
        if blocks == 3:
            downsample = nn.Sequential(
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size , stride= stride,
                          padding=0,dtype=torch.float)
            )
            downsample2 = nn.Sequential(
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size = 3*kernel_size-2 , stride= stride,
                          padding=0,dtype=torch.float))
        if block == ResBlock1d:
            downsample = nn.Sequential(
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=0,dtype=torch.float))
                
            downsample2 = nn.Sequential(
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size = 3*kernel_size-2, stride=stride,
                          padding=0,dtype=torch.float)
            )
        
        if blocks == 4:
            num_conv = 3
            downsample = nn.Sequential(
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3*kernel_size-2 , stride= stride,
                          padding=0,dtype=torch.float)
            )
        for _ in range(blocks):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, downsample=downsample, downsample2= downsample2,num_conv=num_conv))

        return nn.Sequential(*layers)



    def forward(self, x, info=None):
        
        out = x
        out = self.conv(out)  # x => [b, 32, 8, 2476]
        out = self.bn(out)
        out = self.relu(out)
        
        out = self.conv2(out)  # x => [b, 32, 8, 2476]
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)  # x => [b, 32, 8, 2476]
        out = self.bn3(out)
        out = self.relu3(out)
        
        out = self.conv4(out)  # x => [b, 32, 8, 2476]
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0],out.shape[1])
        out = self.fc_1(out)

        return out

class MyDataset(Dataset):

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

def get_all_data(X_train, y_train, batch_size):
    y_tensor = torch.tensor(np.array([y_train]), dtype=torch.int32)
    y_tensor = torch.transpose(y_tensor, 0, 1)
    X_tensor = torch.tensor(np.array([X_train]), dtype=torch.float32)
    X_tensor = torch.transpose(X_tensor, 0, 1)

    _, counts = np.unique(y_train, return_counts=True)
    class_weights = [sum(counts) / c for c in counts]
    example_weights = [class_weights[e] for e in y_train.flatten()]
    sampler = WeightedRandomSampler(example_weights, X_train.shape[0])
    data_loader = DataLoader(MyDataset(X_tensor, y_tensor), batch_size=batch_size) #, sampler=sampler

    return data_loader

def resample(sig, target_point_num=None):
    '''
    resample the original signal
    :param sig: original signal
    :param target_point_num
    :return: resampled signal
    '''
    from scipy import signal
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def transform(sig, train=False):
    '''
    resample the original signal
    :param sig: original signal
    :return: resampled signal in tensor form
    '''
    sig = resample(sig, 2048)
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.double)
    return sig

def read_data(X_train_path, y_train_path, X_test_path, extract_data):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:,1]

    if extract_data:
        train_ids, test_ids = X_train.iloc[:, 0], X_test.iloc[:, 0]
        X_train, X_test = X_train.iloc[:,1:], X_test.iloc[:,1:]
    else:
        train_ids, test_ids = pd.DataFrame(list(range(0, X_train.shape[0]))), pd.DataFrame(list(range(0, X_test.shape[0])))

    return X_train, y_train, train_ids, X_test, test_ids


num_features_avg = 0


def load_data(X_train: str, y_train: str, X_test: str, read_test: bool, read_train: bool):
    start_read = time.time()
    X_train = pd.read_csv(X_train)
   
    X_test = pd.read_csv(X_test)
    X_test = X_test[X_train.columns]

    X_test = X_test.to_numpy() if read_test else None
    X_train = X_train.to_numpy() if read_train else None
   
#     X_test_ind = X_test[:, 0] if read_test else None
    X_test = X_test if read_test else None
    y_train = pd.read_csv(y_train).to_numpy()[:, 1:] if read_train else None

    # TODO fix later to K-Fold
    if read_train:
        tmp=(X_train!=0)
        print("TMP::")
        print(tmp)
        rows = (X_train != 0).sum(1)
        print("ROWSS::")
        print(rows)
        global num_features_avg
        num_features_avg = int(rows.sum() / rows.shape[0])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    else:
        X_train, X_val, y_train, y_val = None, None, None, None

    print(f"Reading data in {time.time() - start_read} seconds.")
    return X_train, y_train, X_val, y_val, X_test


def replace_nan(data, slice_by_avg_len=False):
    start_nan = time.time()
    data[np.isnan(data)] = 0.0
    print(f"Remove nan data in {time.time() - start_nan} seconds.")
    print(data.shape)
    # Trial to cut
    if slice_by_avg_len:
        data = data[:, 0: num_features_avg]
        print(data.shape)
    return data


def preprocess_data(X_train, X_test, y_train):
    X_train = replace_nan(X_train, False)
    X_test = replace_nan(X_test, False)
    y_train = replace_nan(y_train, False)

    return X_train, y_train, X_test


def train(X_train, y_train, X_val, y_val, print_iter=20):
    batch_size = 128
    n_epochs = 80
    lr = 0.01
    best_acc=0
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(device)
    data_loader = get_all_data(X_train, y_train, batch_size)
    X_val = torch.tensor(np.array([X_val]), dtype=torch.float32)
    X_val = torch.transpose(X_val, 0, 1).to(device)

    # model = RNNAttentionModel(1, 64, 'lstm', n_classes=4, kernel_size=4, bidirectional=False).to(device)
    model = SE_ECGNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=5e-6)

    model.train(True)
    best_features=model.state_dict()
    for epoch in range(n_epochs):
        for iteration, (batch_x, batch_y) in enumerate(data_loader):
            # a, b = random.randint(0, 17806 / 2), random.randint(int(17806 / 3), 17806)
            # batch_x = batch_x[:, :, a:(a + b)]
            batch_x = batch_x.to(device)
            # print(batch_x.size())
            batch_y = torch.flatten(batch_y).long().to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
#             free_gpu_cache()       
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
                batch_x = X_val[iteration: 1 + iteration]
                if len(batch_x) > 0:
                    output = model(batch_x)
                    _, pred = torch.max(output.data, 1)
                    preds = pred.cpu().numpy()
                    predictions.extend(preds)

            score_f1=f1_score(y_val, predictions, average='micro')
            print('Eval Epoch / Num epochs: {}/{}....'.format(epoch, n_epochs), end=' ')
            print(" F1 {:.5f}".format(score_f1))
            if score_f1>best_acc :
                print("updated best model")
                best_acc =score_f1
                PATH="models/best_model.pth"
                best_features=copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), PATH)

                
            model.train(True)
        scheduler.step()

    return model,best_features


if __name__ == '__main__':
    X_train,y_train,X_val,y_val,X_test=load_data('./data/train_combined.csv','./data/y_train.csv','./data/test_combined.csv',True,True)
    X_train = replace_nan(X_train, slice_by_avg_len=False)
    X_test = replace_nan(X_test, slice_by_avg_len=False)

    X_val = replace_nan(X_val, slice_by_avg_len=False)
    mm = MinMaxScaler(feature_range=(-1, 1))

    X_train = mm.fit_transform(X_train)
    X_val = mm.transform(X_val)
    model,best_features=train(X_train,y_train,X_val,y_val)
    model_features =SE_ECGNet()
    del model
    import gc 
    gc.collect()
    model_features.load_state_dict(best_features)
    model_features.fc_1=nn.Identity()
    # model_features = model_features.to('cpu')

    X_train_features = pd.read_csv('data/train_combined.csv').to_numpy()
    X_test_features = pd.read_csv('data/train_combined.csv').to_numpy() 
    # X_test_ind_features = X_test[:, 0] 
    # X_test_features = X_test[:, 1:] 

    X_train_features = replace_nan(X_train_features, slice_by_avg_len=False)
    X_test_features = replace_nan(X_test_features, slice_by_avg_len=False)

    # X_val = replace_nan(X_val, slice_by_avg_len=False)
    mm = MinMaxScaler(feature_range=(-1, 1))

    X_train_features = mm.fit_transform(X_train_features)
    X_test_features = mm.transform(X_test_features)
    X_train_features = torch.tensor(np.array([X_train_features]), dtype=torch.float32)
    X_train_features = torch.transpose(X_train_features, 0, 1)
    X_test_features = torch.tensor(np.array([X_test_features]), dtype=torch.float32)
    X_test_features = torch.transpose(X_test_features, 0, 1)

    #extract features
    features_Xtrain=[]
    features_Xtest=[]
    with torch.no_grad():
        model_features.eval()
        for iteration in range(X_train_features.shape[0]):
            batch_x = X_train_features[iteration: 1 + iteration]
            if len(batch_x) > 0:
                output = model_features(batch_x)
    #             _, feat = torch.max(output.data, 1)
                feats = output.cpu().numpy()
    #             print(output)
                features_Xtrain.extend(feats)

        for iteration in range(X_test_features.shape[0]):
            batch_x = X_test_features[iteration: 1 + iteration]
            if len(batch_x) > 0:
                output = model_features(batch_x)
    #             _, feat = torch.max(output.data, 1)
                feats = output.cpu().numpy()
                features_Xtest.extend(feats)
    features_Xtrain = np.asarray(features_Xtrain)
    features_Xtest = np.asarray(features_Xtest)
    np.savetxt("data/X_train_cnn_64_best.csv",features_Xtrain,delimiter = ",")

    np.savetxt("data/X_test_cnn_64_best.csv",features_Xtest,delimiter = ",")