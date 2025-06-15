import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools

data = pd.read_csv("league_of_legends_data_large.csv")
data.head(5)

X, y =  data.drop(['win'], axis=1), data.loc[:,['win']]

print(X.iloc[0])
print("------------------")
print(y.iloc[0])


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(y_test.shape)
print(y_train.shape)

# Convert to tensor
X_train = torch.tensor(X_train_sc, dtype=torch.float32)
X_test = torch.tensor(X_test_sc, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

# Create DataLoader for training and test sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check class balance
labels = torch.cat([y for _, y in train_loader])
print("1:", labels.mean().item())

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x


model = LogisticRegressionModel(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def model_trainer(model, epochs, optimizer, criterion):

    for epoch in range(1, epochs + 1):
        model.train()

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}")

def evaluate(model, loader, name="Test"):
    model.eval()
    acc = 0
    total = 0
    loss_total = 0

    with torch.no_grad():
        for X, y in loader:
            outputs = model(X).detach()
            loss_total += criterion(outputs, y).item() * X.size(0)
            total += y.numel()

    acc += ((outputs > 0.5).float() == y).float().mean()
    avg_loss = loss_total / total
    print(f"{name} Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}")


model_trainer(model, 1000, optimizer, criterion)

evaluate(model, train_loader, name="Train")
evaluate(model, test_loader, name="Test")

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
model_trainer(model, 1000, optimizer, criterion)
evaluate(model, train_loader, name="Train")
evaluate(model, test_loader, name="Test")


# Visualize the confusion matrix
#Change the variable names as used in your code

y_pred_test = model(X_test)

y_pred_test_np = y_pred_test.detach().numpy()
y_pred_test_labels = (y_pred_test_np > 0.5).astype(int)

y_test_np = y_test.detach().numpy().astype(int)
cm = confusion_matrix(y_test_np, y_pred_test_labels)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(2)
plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)
plt.yticks(tick_marks, ['Loss', 'Win'])

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_test_np, y_pred_test_labels, target_names=['Loss', 'Win']))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test_np, y_pred_test_np)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Save the model
torch.save
torch.save(model.state_dict(), "LR_model.pth")
# Load the model
model = LogisticRegressionModel(input_dim=8)
model.load_state_dict(torch.load('LR_model.pth'))
# Evaluate the loaded model
evaluate(model, test_loader, name="Test")

# Hyperparameter tuning
def hyper_param_tuner(epochs=50, criterion=criterion, LRs=[0.001, 0.01, 0.05, 0.1]):
    results = []
    final_loss = 0
    for lr in LRs:
        model = LogisticRegressionModel(8)
        model.load_state_dict(torch.load('LR_model.pth'))
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X.size(0)  # summert loss

        final_loss = running_loss / len(train_dataset)
        results.append((lr, final_loss))
    results.sort(key=lambda x: x[1], reverse=True)
    for lr, acc in results:
        print(f"LR={lr:.3f} --> Loss={final_loss:.4f}")

# Run hyperparameter tuning
hyper_param_tuner()

# Extract the weights of the linear layer
FI = model.linear.weight.data.numpy().flatten()
feature_names = X.columns
df = pd.DataFrame({'Feature': feature_names,
                'learned_weights': FI
                })
df = df.reindex(df.learned_weights.abs().sort_values(ascending=False).index)
# Create a DataFrame for feature importance
df.head(5)
## Write your code here
plt.bar(df.Feature, df.learned_weights)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()