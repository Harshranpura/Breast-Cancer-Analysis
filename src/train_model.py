import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")
X = df.iloc[:, :-1].values  # Features
y = df["label"].values  # Target (0: Malignant, 1: Benign)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

optimizers = ["SGD", "Adam", "RMSprop"]
results = {}

for name in optimizers:
    print(f"\n🔹 Training with {name} optimizer")
    model = NeuralNet(input_size=X.shape[1], hidden_size=16, output_size=2)

    # Assign optimizer dynamically
    if name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()
    loss_values = []

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    results[name] = loss_values

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

for name, loss in results.items():
    plt.plot(range(1, 51), loss, label=name)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Comparison for Different Optimizers")
plt.legend()
plt.show()

# Test model accuracy
with torch.no_grad():
    test_outputs = model(X_test)
    _, predictions = torch.max(test_outputs, 1)
    accuracy = (predictions == y_test).sum().item() / y_test.size(0)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Select 5 random test samples
import random

indices = random.sample(range(len(X_test)), 5)
sample_features = X_test[indices]
sample_labels = y_test[indices]

# Predict using the trained model
with torch.no_grad():
    sample_outputs = model(sample_features)
    _, sample_predictions = torch.max(sample_outputs, 1)

# Print results
for i in range(len(indices)):
    print(f"Actual: {'Malignant' if sample_labels[i] == 0 else 'Benign'}, Predicted: {'Malignant' if sample_predictions[i] == 0 else 'Benign'}")
