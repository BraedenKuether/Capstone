# OSIG Portfolio Analysis Software
This repo contains the code for the OSIG portfolio analysis website which you can visit [here]("").
The website uses django for the backend and react for the frontend and graphing. The ML model is in python and leverages pyTorch. The financial data is provided through the IEX cloud api and you can create a free account [here](https://www.iexcloud.io/core-data/).

The website code is located in the backend folder and the machine learning model is located in the [portfolio analysis](https://github.com/BraedenKuether/Capstone/tree/main/backend/osig/portfolio_analysis) folder under MLUtils.


# Django Requirements
* [mysqlclient](https://pypi.org/project/mysqlclient/)
* [REST framework](https://www.django-rest-framework.org/#installation)
  
# React Requirements
* run npm install

# Python Requirements
* [pyTorch](https://pytorch.org/get-started/locally/#windows-installation)
* [matplotlib](https://matplotlib.org/stable/users/installing.html)
* [scikitlearn](https://scikit-learn.org/stable/install.html)
* [pyEx](https://pyex.readthedocs.io/en/latest/)

# Machine Learning Documentation
* [Running a Model Example](#running-a-model-example)
* [Connection to Django Backend](#connection-to-django-backend)
* [Scripts](#scripts)
  * [Portfolio.py](#portfoliopy)
  * [Tester.py](#testerpy)
  * [Trainer.py](#trainerpy)
  
## Scripts
All the following scripts are located in [backend/osig/portfolio_analysis/MLUtils](https://github.com/BraedenKuether/Capstone/tree/main/backend/osig/portfolio_analysis/MLUtils) folder.

## Running a Model Example
Here is a simple of running the model on three tickers: f (Ford), fb (facebook), Alphabet (googl). This example uses sandbox mode so that the it does not use up API calls.
```
import pyEX as px
from .MLUtils import Portfolio as P
from .MLUtils import Tester as T
from .MLUtils.Trainer import *

IEX_TOKEN = "{INSERT_TOKEN_HERE}"
client = px.Client(IEX_TOKEN, version="sandbox")

tickers=['f','fb','googl']
p = P.Portfolio(tickers,client,earnings=True)
user_environment = T.Tester(p,10,60,train_func = train_net_earnings)
predictions = user_environment.trainModel()
print(predictions)
```

## Connection to Django Backend
The website uses the machine learning tools and the Portfolio analyser to calculate several things that are displayed on each run. All of these are handled in the [backend\osig\portfolio_analysis\views.py](https://github.com/BraedenKuether/Capstone/blob/main/backend/osig/portfolio_analysis/views.py) file. The create_run function calculates all the neccessary data when a user submits a run from the front-end.

### Portfolio.py
The portfolio constructor takes in 3 arguments:
* assets
  * array of strings representing each ticker symbol
* client
  * the pyEX client 
* earnings
*   True/False. Setting it to True uses the Earnings ML model, which incorporates financial data from each ticker, while setting it false uses only price history. 

The Portfolio object is passed into the Tester object in order to run the model.

### Tester.py
The Tester object takes in 6 arguments:
* P
  * The portfolio object which contains the data for the previously entered tickers.
* timePeriod 
  * How far in advance you want the model to predict the weights for. For example, after setting timePeriod to 30 days and training the model, the recommended weights are supposed to be used for up to 30 days. It is then recommended to retrain the model once the 30 days up.
* batchSize
  *  Number of data points per batch. For example, having timePeriod set to 30 and batchSize to 5 will have each batch contain 5 sets of consecutive 30-days of ticker prices. Note that this is a sliding batch, so the second data point in the batch start on day 2 from the first data point.
* train_func
  * Which training function to use from MLUtils.Trainer. Can either be train_net or train_net_earnings. Default is train_net, which traing only using ticker prices. train_net_earnings uses a different model which incorporates financial data in parallel linear layers.
* test_length
  * Default 126. The length of the testing dataset. The model excludes the last test_length days from the dataset in order to evaluate the performance in the validation_set or validation_set_earnings function.
* epochs
  * Defaults to 100. The number of epochs to train the model for.

The Tester class also has several different functions which performs statistical analyses on the inputted portfolio. Some of these functions requires you to set the weights use the setWeights function, which is an array that has same length as the tickers array. These weights are useful to investors in case they want to put in how they weigh each asset so that they can calculate things like the total performance or the sharpe ratio. These weights do not affect the neural net training process.

### Trainer.py
This file contains the functions for both training and testing the model. Some notable functions are:
* train_net_earnings
  * Trains a neural network on asset history and incorporate financial data
* train_net
  * Train a neural network only on asset history.
* validation_set_earnings
  * Simulates returns on a neural network trained with earnings info on a price history. Returns graph of performance simulation where 1 is the starting value. EX: A performance history where the last y value is 1.5 means the model gained 50% in the simulation.
* validation_set
  * Same as validation set earnings, but only works with on a neural net trained with train_net

