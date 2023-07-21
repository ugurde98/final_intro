from warnings import filterwarnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def matrixMetrics(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f2_measure = fbeta_score(y_test, y_pred, average='macro', beta=2.0)
    return precision, recall, f2_measure

filterwarnings('ignore')

lb = pd.read_csv('labels.csv')
lb = lb.drop(labels=['Sample'], axis=1)

df = pd.read_csv('data.csv')
df = df.drop(df.columns[0], axis='columns')

# bağımlı ve bağımsız değişkenler belirlenmiştir.
y = lb['disease_type']
X = df.iloc[:, 1:]

# Veri seti %70 train, %30 test olarak bölünmüştür.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Verileri normalize edelim:
X_train_normalized = X_train / X_train.sum(axis=1).values[:, None]
X_test_normalized = X_test / X_test.sum(axis=1).values[:, None]

# Convert labels to numeric values using LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Verileri PyTorch Tensor'larına dönüştürelim
X_train_tensor = torch.tensor(X_train_normalized.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_normalized.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

# DataLoader'ları oluşturalım
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# PyTorch ile Çok Katmanlı Perceptron'u (MLP) tanımlayalım
class MLP(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(MLP, self)._init_()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.conv1x1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Veriyi (batch_size, hidden_size, 1) boyutunda 1x1 filtreli bir özellik haritasına çevirelim
        x = x.unsqueeze(2)  # Veriyi (batch_size, hidden_size, 1) boyutuna çevirme
        x = self.conv1x1(x)
        x = x.squeeze(2)  # Boyutu tekrar (batch_size, hidden_size) yapma
        x = self.fc2(x)
        return x

# Modeli oluşturalım ve eğitelim
def MultilayerPerceptron():
    input_size = X_train_normalized.shape[1]
    hidden_size = 64
    output_size = len(lb['disease_type'].unique())

    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

    model.eval()
    with torch.no_grad():
        y_pred = []
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())

    # Inverse transform the predicted labels
    y_pred = label_encoder.inverse_transform(y_pred)

    precision, recall, f2_measure = matrixMetrics(y_test, y_pred)

    print("-----------------------Multilayer Perceptron Algorithm Result Start------------------------")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F2 Score:", f2_measure)
    print("-----------------------Multilayer Perceptron Algorithm Result End------------------------")


print("xxxxx")
MultilayerPerceptron()  # PyTorch MLP
print("xxxxx")