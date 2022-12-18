# import dependencies
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets


class data_set(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, features, labels):
        'Initialization'
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)
        self.num_sample = labels.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index]


    def __len__(self):
        'Denotes the total number of samples'
        return self.num_sample


def split_data(X, y, test_size):
    """
    return the splitted data based on test size

    parameters
    ------------------
    X : input data array
    Y : input data array
    stratify : split data proportional to a feature
    test_size : the proportion of the dataset to include in the test split
    
    Output
    ------------------
    List containing train-test split of inputs
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    return X_train, X_test, y_train, y_test


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(LogisticRegressionModel, self).__init__()
        self.Linear = nn.Linear(input_feature, output_feature)

    def forward(self, feature):
        y_predict = torch.sigmoid(self.Linear(feature))
        return y_predict


if __name__=="__main__" :

    # Dataset & Split
    data = datasets.breast_cancer()
    X_train, X_test, y_train, y_test = split_data(data.data, data.target, 0.2)

    # scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Generators train & test data
    params = {  'batch_size':32,
                'shffle' : True,
                'num_workers' : -1 }

    train_data = data_set(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, **params)

    test_data = data_set(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, **params)

    # check data
    print("First iteration of train data :\n")
    print(data_iter.next(iter(train_dataloader)))
    print("Length of train data :\n", len(train_dataloader))

    print("First iteration of test data :\n")
    print(data_iter.next(iter(test_dataloader)))
    print("Length of train data :\n", len(test_dataloader))


    # model -----------------------
    model = LogisticRegressionModel(n_features, 1)

    # loss and optimizer
    Learning_rate = 0.01
    Loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

    # training loop
    n_epochs = 100

    for epoch in range(n_epochs):
        for inputs, labels in train_dataloader:
            # forward and loss
            y_predict = model(inputs)
            loss = Loss(y_predict, labels)

            # backward
            loss.backward()

            # update weight
            optimizer.step()

            # reset gradient
            optimizer.zero_grad()


        # show result
        print(f'epochs {epoch+1} , loss {loss.item():.4f}')

    with torch.no_grad():
        for inputs, labels in test_dataloader: