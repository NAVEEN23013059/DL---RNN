# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

## Regeister number : 212223240106
## name : Naveen.s


## DESIGN STEPS
### STEP 1: 

Write your own steps

### STEP 2: 



### STEP 3: 



### STEP 4: 



### STEP 5: 



### STEP 6: 





## PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv(r'C:\Users\admin\Downloads\Salary.csv')

print(df.head())
print(df.columns)

X = df[['Level']].values
y = df[['Salary']].values

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

X_tensor = torch.tensor(
    X_scaled,
    dtype=torch.float32
).unsqueeze(1)

y_tensor = torch.tensor(
    y_scaled,
    dtype=torch.float32
)

dataset = TensorDataset(
    X_tensor,
    y_tensor
)

train_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

class RNNModel(nn.Module):

    def __init__(self):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(50, 1)

    def forward(self, x):

        out, hidden = self.rnn(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = RNNModel()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("Device:", device)
print(model)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

num_epochs = 100

train_losses = []

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    for data, labels in train_loader:

        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(data)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss / len(train_loader)
    )

    train_losses.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Training Loss: {epoch_loss:.6f}"
        )

print("Name: Naveen")
print("Register Number: 212223240106")

plt.figure(figsize=(10, 6))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()


model.eval()

with torch.no_grad():

    predicted = model(
        X_tensor.to(device)
    ).cpu().numpy()

predicted_salary = y_scaler.inverse_transform(
    predicted
)

actual_salary = y_scaler.inverse_transform(
    y_scaled
)

print("Predicted Salary:")
print(predicted_salary)

print("\nActual Salary:")
print(actual_salary)

plt.figure(figsize=(10, 6))

plt.plot(
    actual_salary,
    marker='o',
    label="Actual Salary"
)

plt.plot(
    predicted_salary,
    marker='x',
    label="Predicted Salary"
)

plt.xlabel("Employee Level")
plt.ylabel("Salary")

plt.title(
    "Salary Prediction using RNN"
)

plt.legend()
plt.show()

# Final prediction
print(
    f"\nPredicted Salary: "
    f"{predicted_salary[-1][0]:.2f}"
)

print(
    f"Actual Salary: "
    f"{actual_salary[-1][0]:.2f}"
)

```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="448" height="248" alt="image" src="https://github.com/user-attachments/assets/9eb1a853-c066-47f1-a55c-beb72ca8a990" />

<img width="1062" height="681" alt="image" src="https://github.com/user-attachments/assets/da443223-fe65-49f5-8c1e-82fda43f434a" />


## True Stock Price, Predicted Stock Price vs time

<img width="267" height="555" alt="image" src="https://github.com/user-attachments/assets/9090f51c-7e38-41c7-8b80-ee1570a1aad4" />

<img width="1056" height="681" alt="image" src="https://github.com/user-attachments/assets/a21fdde8-97ec-4b7a-aaee-a34eff425d4b" />

### Predictions

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
