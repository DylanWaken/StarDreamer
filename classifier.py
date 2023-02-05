import torch
import torchvision

import numpy as np
import h5py

# To get the images and labels from file
with h5py.File('./dataset/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images'], dtype=np.float32)
    labels = np.array(F['ans'], dtype=np.float32)

# NHWC to NCHW
images = np.transpose(images, (0, 3, 1, 2))

print(labels)

# put the images and labels into a torch dataset
dataset = torch.utils.data.TensorDataset(torch.from_numpy(images / 255), torch.from_numpy(labels))

# split the dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# put the train and test dataset into a dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# define the model
model = torchvision.models.resnet18(pretrained=True).to("cuda")
criterion = torch.nn.CrossEntropyLoss()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
model.fc.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

n_epochs = 30
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(train_loader)
for epoch in range(1, n_epochs + 1):

    running_loss = 0.0
    correct = 0
    total = 0

    print(f'Epoch {epoch}\n')

    for batch_idx, (data_, target_) in enumerate(train_loader):
        data_, target_ = data_.to("cuda"), target_.to("cuda")
        optimizer.zero_grad()

        outputs = model(data_)
        loss = criterion(outputs, target_.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred == target_).item()
        total += target_.size(0)
        if (batch_idx) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

        if(loss.item() < 0.3):
            torch.save(model.state_dict(), 'weights/classifier.pt')
            exit(0)
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
    batch_loss = 0
    total_t = 0
    correct_t = 0

    with torch.no_grad():
        for data_t, target_t in (test_loader):
            data_t, target_t = data_t.to("cuda"), target_t.to("cuda")
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t.long())
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t == target_t).item()
            total_t += target_t.size(0)

        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(test_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'weights/classifier.pt')
            print('Improvement-Detected, save-model')
