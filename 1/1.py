#import dependencies
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE, r2_score as R2

# create data
def create_data(n_samples, n_features, noise, bias):
    """
    create a dataset based for linear regression problem

    parameters
    -------------------
    n_samples : Number of sample to create dataset
    n_features : Number of features that consider for each sample
    noise : The standard deviation of the gaussian noise applied to the output
    random_state : Determines random number generation for dataset creation

    Output
    -------------------
    X : The input samples
    Y : The output values
    """
    X, Y = datasets.make_regression(n_samples=n_samples, n_features=n_features,
                                    random_state=0, noise=noise, bias=bias)


    X = torch.from_numpy(X.astype(np.float32))
    Y = torch.from_numpy(Y.astype(np.float32))
    
    # reshape y into vector state
    Y = Y.view(-1, 1)
    
    return X, Y


def split_data(X, Y, test_size=0.2):
    """
    return the splitted data based on test size

    parameters
    ------------------
    X : input data array
    Y : input data array
    test_size : the proportion of the dataset to include in the test split
    
    Output
    ------------------
    List containing train-test split of inputs
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                test_size=test_size, shuffle=True)

    return X_train, X_test, Y_train, Y_test


def evaluate(y, y_pred):
    """
    evaluate the predicted values

    parameter
    -------------------
    y : the true value of sample
    y_pred : value prediction of sample

    Output
    -------------------
    print evaluation based on four criteria
    """

    mse = MSE(y, y_pred)
    rmse = np.sqrt(mse)
    mae = MAE(y, y_pred)
    r2 = R2(y, y_pred)

    print("mean squared error and rmse): {}, {}".format(mse, rmse))
    print(f'mean absolute error (rmse): {mae}')
    print("mean r2 score : {}".format(r2))



if __name__=="__main__":
    
    # Create dataset
    n_samples, n_features =300, 1
    noise, bias = 20.0, 100.0
    X, Y = create_data(n_samples, n_features, noise, bias)

    # Split data
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=0.2)
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    n_samples, n_features = X_train.shape


    # Create model
    input_size, output_size = n_features, 1
    model = nn.Linear(input_size, output_size)

    # Define loss and optimizer
    Learning_rate = 0.01
    Loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

    # Training loop
    n_epoch = 100

    for epoch in range(n_epoch):
        # Forward and loss
        y_predict = model(X_train)
        loss = Loss(y_predict, Y_train)

        # Backward
        loss.backward()

        # Update weight
        optimizer.step()

        # Reset gradient value for next iteration
        optimizer.zero_grad()

        # Show result each number of epoch
        if ((epoch+1)%10 == 0):
            print(f'epochs {epoch+1} , loss {loss.item():.4f}')


    y_pred = model(X_test).detach().numpy()
    evaluate(Y_test, y_pred)

    
    # Plot
    predicted = model(X).detach().numpy() # detach to exit from calculate grad
    plt.plot(X, Y, 'ro')
    plt.plot(X, predicted, 'b-')
    plt.show()






    
    
    
