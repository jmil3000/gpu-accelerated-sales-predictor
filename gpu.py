import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from dotenv import load_dotenv
import os

file_path = os.getenv('DATASET')
walmart_data = pd.read_csv(file_path)
walmart_data['Date'] = pd.to_datetime(walmart_data['Date'], format='%d-%m-%Y')
walmart_data['Year'] = walmart_data['Date'].dt.year
walmart_data['Month'] = walmart_data['Date'].dt.month
walmart_data['Day'] = walmart_data['Date'].dt.day
walmart_data.drop('Date', axis=1, inplace=True)

#check nans
missing_values = walmart_data.isnull().sum()
numerical_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Day']
scaler = StandardScaler()
walmart_data[numerical_cols] = scaler.fit_transform(walmart_data[numerical_cols])
walmart_data = pd.get_dummies(walmart_data, columns=['Store'])
preprocessed_head = walmart_data.head()

print("Missing Values:\n", missing_values)
print("\nFirst Few Rows of Preprocessed Data:\n", preprocessed_head)

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

X = walmart_data.drop('Weekly_Sales', axis=1)
y = walmart_data['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
y_train_tensor = y_train_tensor.view(y_train_tensor.shape[0], 1)
y_test_tensor = y_test_tensor.view(y_test_tensor.shape[0], 1)

class FNN(nn.Module):
    def __init__(self, input_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) 
        self.fc2 = nn.Linear(128, 64)         
        self.fc3 = nn.Linear(64, 1)          
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
model = FNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
