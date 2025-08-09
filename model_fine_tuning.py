import torch
import timm
import os
from data_preprocessing import prepare_datasets, get_dataloaders
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


def train(model, train_loader, val_loader, epochs, criterion, scaler, optimizer, scheduler=None, device='cpu'):
    """
    Обучение модели
    :param model: модель
    :param train_loader: загрузчик для обучающей выборки
    :param val_loader: загрузчик для валидационной выборки
    :param epochs: сколько эпох будет обучаться модель
    :param criterion: функция потерь
    :param scaler: скейлер для предотвращения размытия градиентов
    :param optimizer: оптимизатор
    :param scheduler: Шэдулер для постепенного снижения параметра обучения
    :param device: где, будет обучаться модель
    :return: tuple(all_loses, val_loses, accuracies, best_model): списки с значениями функций потерь на обучающем и
    валидационном наборах, список значений точностей на вал. наборе, и state_dict() лучшей на вал. наборе модели
    """
    all_loses = []
    val_loses = []
    accuracies = []
    best_model = model.state_dict()
    best_model_acc = 0
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.long().to(device=device)
            output = model(inputs)
            loss = criterion(output, targets)
            scaler.scale(loss).backward()
            all_loses.append(loss.item())
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()

        training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        model.eval()
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.long().to(device)
            loss = criterion(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)

        valid_loss /= len(val_loader.dataset)
        val_loses.append(valid_loss)
        acc = accuracy_score(targets.cpu(),
                             torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1).long().cpu().detach())
        accuracies.append(acc)
        if acc > best_model_acc:
            best_model_acc = acc
            best_model = model.state_dict()
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch,
                                                                                                    training_loss,
                                                                                                    valid_loss, acc))
    return all_loses, val_loses, accuracies, best_model


def train_stats(all_loses, val_loses, accuracies, epochs):
    """
    Рисует графики, по которым можно отслеживать обучение
    :param all_loses: Значения функции потерь на обучающей выборке
    :param val_loses: Значения функции потерь на валидационной выборке
    :param accuracies: Значения точности (accuracy) на валидационной выборке
    :param epochs: Число эпох, сколько обучалась модель
    :return: None
    """
    plt.plot(np.arange(0, len(all_loses)), all_loses, label='Training Loss')
    plt.title('Training Loss with Cross-Entropy per batch')
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, len(all_loses)+1, 10))
    plt.legend(loc='best')
    plt.show()
    # plot the training loses
    plt.plot(np.arange(1, epochs+1), val_loses, label='Validation Loss')
    plt.title('Validation Loss with Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, epochs+1, 2))
    plt.legend(loc='best')
    plt.show()
    plt.plot(np.arange(1, epochs+1), accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy with Cross-Entropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, epochs+1, 1))
    plt.legend(loc='best')
    plt.show()


def model_test(model, test_dataset, test_loader, device='cpu'):
    """
    Запуск модели модели на тестовом наборе
    :param model: классификатор
    :param test_dataset: тестовый датасет
    :param test_loader: загрузчик данных на тесте
    :param device: устройство, где будут вычисления
    :return: точность (accuracy) модели на тестовом наборе
    """
    model.eval()
    all_preds = torch.Tensor([])
    all_preds = all_preds.to(device)
    for batch in test_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        model = model.to(device)
        predictions = torch.argmax(torch.nn.functional.softmax(model(inputs), dim=1), dim=1)
        all_preds = torch.cat((all_preds, predictions))
    all_preds = all_preds.cpu().detach().numpy()
    accuracy = accuracy_score(np.array(test_dataset.labels), all_preds)
    print('Test Accuracy: ', accuracy)
    return accuracy


if __name__ == "__main__":
    model = timm.create_model('davit_base', pretrained=True)
    path_to_data = os.path.join('D:', 'test_case_data')
    train_dataset, val_dataset, test_dataset, c_to_i = prepare_datasets(path_to_data, stats=True)

    data_point = train_dataset[0][0]
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset)

    model.head.fc = torch.nn.Linear(in_features=1024, out_features=2, bias=True)

    acc = model_test(model, test_dataset, test_loader)
