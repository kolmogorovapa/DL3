from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='Dropout added')

train_indices = torch.arange(50000)

train_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=transforms.ToTensor(), train=True)
train_cifar_dataset = data_utils.Subset(train_cifar_dataset, train_indices)

test_indices = torch.arange(10000)
test_cifar_dataset = datasets.CIFAR100(download=True, root='./', transform=transforms.ToTensor(), train=False)
test_cifar_dataset = data_utils.Subset(test_cifar_dataset, test_indices)

train_cifar_dataloader = DataLoader(dataset=train_cifar_dataset, batch_size=1, shuffle=True)
test_cifar_dataloader = DataLoader(dataset=test_cifar_dataset, batch_size=1, shuffle=True)


class CIFAR100PredictorPerceptron(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        x = self.softmax(x)

        return x


model = CIFAR100PredictorPerceptron(input_size=3072, hidden_size=180, output_size=100)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 20

for epoch in range(num_epochs):
    train_error = 0
    train_correct_guess = 0
    for x, y in train_cifar_dataloader:
        model.train()
        optimizer.zero_grad()
        prediction = model(x)
        zero_tensor = torch.zeros_like(prediction)
        train_loss = loss_fn(prediction, y)
        train_error += train_loss.item()

        predicted_indices = torch.argmax(prediction, dim=1)
        train_correct_guess += (predicted_indices == y).sum().item()

        train_loss.backward()
        optimizer.step()

    writer.add_scalar('Train Loss', train_error / len(train_cifar_dataset), epoch)
    writer.add_scalar('Train Accuracy', train_correct_guess / len(train_cifar_dataset), epoch)

    test_correct_guess = 0
    test_error = 0
    for x, y in test_cifar_dataloader:
        model.eval()
        prediction = model(x)
        loss_test = loss_fn(prediction, y)
        test_error += loss_test.item()

        predicted_indices = torch.argmax(prediction, dim=1)
        test_correct_guess += (predicted_indices == y).sum().item()

    writer.add_scalar('Test Loss', test_error / len(test_cifar_dataset), epoch)
    writer.add_scalar('Test Accuracy', test_correct_guess / len(test_cifar_dataset), epoch)

writer.close()