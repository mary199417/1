# import dependencies
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# Evaluation
from sklearn.metrics import classification_report, roc_auc_score 
from sklearn.metrics import confusion_matrix, accuracy_score


class data_set(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, features, labels):
        'Initialization'
        self.X = torch.from_numpy(features.astype(np.float32))
        self.y = (torch.from_numpy(labels.astype(np.float32))).view(-1,1)
        self.num_sample = labels.shape[0]
        # print(self.X.shape, self.y.shape)


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


def evaluate(y, y_pred):
    """
    evaluate the predicted values

    parameter
    -------------------
    y : the true value of sample
    y_pred : value prediction of sample

    Output
    -------------------
    print evaluation based on multiple criteria
    """
    cl_rp = classification_report(y, y_pred)
    auc = np.round(roc_auc_score(y, y_pred), 3)
    accuracy = np.round(accuracy_score(y, y_pred), 3)
    conf_mat = confusion_matrix(y, y_pred)

    print(cl_rp)
    print(f'Auc : {auc}')
    print(f'accuracy : {accuracy}')
    print("confusion_matrix :\n", conf_mat)


if __name__=="__main__" :

    # Dataset & Split
    data = datasets.load_breast_cancer()
    X_train, X_test, y_train, y_test = split_data(data.data, data.target, 0.2)


    # scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    # Generators train & test data
    params = {  'batch_size':32,
                'shuffle' : True,
                'num_workers' : 3 }

    train_data = data_set(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, **params)

    # check data
    # print("First iteration of train data :\n")
    # print((iter(train_dataloader).next()))
    # print("Length of train data :\n", len(train_dataloader))



    # model -----------------------
    model = LogisticRegressionModel(X_train.shape[1], 1)

    # loss and optimizer
    Learning_rate = 0.01
    Loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

    # training loop
    n_epochs = 3

    for epoch in range(n_epochs):
        for inputs, labels in train_dataloader:
            # forward and loss
            y_predict = model(inputs)
            # print(inputs.shape, labels.shape, y_predict.shape)
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
        print(type(X_test), type(y_test))
        y_predicted = model(X_test)
        y_predicted = y_predicted.round()
    evaluate(y_test, y_predicted)