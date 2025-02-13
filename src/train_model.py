import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from neural_net import NeuralNet

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")
X = df.iloc[:, :-1].values  # Features
y = df["label"].values  # Labels

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
input_size = X.shape[1]
model = NeuralNet(input_size=input_size, hidden_size=16, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 50
loss_values = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("✅ Model training completed!")

# Evaluate accuracy
with torch.no_grad():
    test_outputs = model(X_test)
    _, predictions = torch.max(test_outputs, 1)
    accuracy = (predictions == y_test).sum().item() / y_test.size(0)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
