import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class CustomImageDataset(Dataset):
    """
    Датасет для изобрвжений с предустановленной предобработкой.
    """
    def __init__(self, image_paths, labels, transform=None, is_train=True):
        """
        Args:
            image_paths (list): Список путей к точкам данных
            labels (list): Метки
            transform (callable, optional): Oпциональный пайплайн предобработки
            is_train (bool): Обучающая ли это выборка (влияет на аугментацию)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train
        self.transform = transform or self.get_default_transform()

    def get_default_transform(self):
        """Трансформации полученных изображений, в частности:
        Для тренировочного набора:
            Переформатирование (Resize),
            Отражение по горизонтали с вероятностью 1/3
            Отражение по вертикали с вер-тью 1/3
            Изменение цвета и нормализация

        Для валидационного и тестового набора:
            Переформатирование и нормализация

        """
        if self.is_train:
            return transforms.Compose([
                transforms.Resize((384, 448)),
                transforms.RandomHorizontalFlip(p=0.33),
                transforms.RandomVerticalFlip(p=0.33),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0]),

            ])
        else:
            return transforms.Compose([
                transforms.Resize((384, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.randn(3, 224, 224), -1


def prepare_datasets(data_dir, test_size=0.2, val_size=0.1, random_state=42, stats=False,
                     transform=None, return_counts=False):
    """
    Подготовка обучающего, валидационного и тестового датасетов из файлов.

    Необходимая структура файлов:
    data_dir/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
        ...

    Args:
        data_dir (str): Путь к корневой директории
        test_size (float): Доля точек в тестовом наборе
        val_size (float): Доля точек в валидационном наборе (от оставшегося после выделения тестового набора)
        random_state (int): Random seed для воспроизводимости эксперимента
        stats (bool): писать ли статистики по датасету
        transform (Callable): набор преобразований изображения
        return_counts (bool): возвращать ли число точек в каждом классе

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_to_idx, return_counts)
    """
    # Gather all image paths and labels
    classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    image_paths = []
    labels = []

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        cls_idx = class_to_idx[cls_name]

        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_dir, img_name)
                image_paths.append(img_path)
                labels.append(cls_idx)

    _, label_counts = np.unique(labels, return_counts=True)

    # Split into train, val, test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state,
        stratify=labels
    )

    # Further split train into train and val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=random_state,
        stratify=train_labels
    )

    # Create datasets
    train_dataset = CustomImageDataset(train_paths, train_labels, transform, is_train=True)
    val_dataset = CustomImageDataset(val_paths, val_labels, is_train=False)
    test_dataset = CustomImageDataset(test_paths, test_labels, is_train=False)

    if stats:
        print(f'There are {len(class_to_idx)} classes in the dataset.')
        print(f' There are {len(train_labels)} samples in the train dataset')
        print(f' There are {len(val_labels)} in the validation dataset')
        print(f' There are {len(test_labels)} in the test dataset')
        values, counts = np.unique(np.array(labels), return_counts=True)
        print(f' Class ratio: {counts[0] / counts[1]}')

    if return_counts:
        return train_dataset, val_dataset, test_dataset, class_to_idx, label_counts

    return train_dataset, val_dataset, test_dataset, class_to_idx


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """
    Создание DataLoader.

    Args:
        train_dataset, val_dataset, test_dataset: Dataset
        batch_size (int): Batch size для даталоадеров

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    path_to_data = os.path.join('D:', 'test_case_data')

    train, val, test, c_to_i = prepare_datasets(path_to_data)

    print(train)

    print(c_to_i)
