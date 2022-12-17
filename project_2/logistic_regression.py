# import dependencies
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score 
from sklearn.metrics import confusion_matrix, accuracy_score

def create_dataset():
    """
    Load a dataset based for Logistic regression problem    

    Output
    -------------------
    X : The input samples
    y : The output values
    """
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    return X, y

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



if __name__=="__main__":
    X, y = create_dataset()
    n_sample, n_features = X.shape

    # split data to train and test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # scale data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # convert numpy array to tensor
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    # reshape y
    y_train = y_train.view(-1, 1)
    y_test = y_test.view(-1,1)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # model
    model = LogisticRegressionModel(n_features, 1)

    # loss and optimizer
    Learning_rate = 0.01
    Loss = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

    # training loop
    n_epochs = 100

    for epoch in range(n_epochs):
        # forward and loss
        y_predict = model(X_train)
        loss = Loss(y_predict, y_train)

        # backward
        loss.backward()

        # update weight
        optimizer.step()

        # reset gradient
        optimizer.zero_grad()

        # show result
        if ((epoch+1)%10==0):
            print(f'epochs {epoch+1} , loss {loss.item():.4f}')

    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted = y_predicted.round()
    evaluate(y_test, y_predicted)
    
