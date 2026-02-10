# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

### Name:SWETHA C
### Register Number:212224230283

```
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(nethraa_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = nethraa_brain(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
```
## Dataset Information
<img width="196" height="351" alt="image" src="https://github.com/user-attachments/assets/c18d0d45-819b-472d-860a-f94eb0bcd284" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="686" height="555" alt="image" src="https://github.com/user-attachments/assets/723794b7-6e82-4ac1-808f-17c8e4c8fecc" />


### New Sample Data Prediction

<img width="793" height="400" alt="image" src="https://github.com/user-attachments/assets/af563207-0463-4571-ac24-0b4e45843727" />
<img width="793" height="400" alt="image" src="https://github.com/user-attachments/assets/d48e7612-780e-4fb4-b70f-b05ef8ec1bbe" />



## RESULT
The neural network regression model was successfully developed and trained.
