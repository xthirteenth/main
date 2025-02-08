from src.models.cnn import SimpleCNN
from src.train import train_model
from src.utils import load_config, set_seed, get_device
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def main():
    # Загрузка конфигурации
    config = load_config()
    
    # Установка случайности
    set_seed(config['random_seed'])
    
    # Выбор устройства
    device = get_device(config)
    
    # Подготовка трансформаций и данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False
    )
    
    # Создание модели
    model = SimpleCNN(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    # Настройка оптимизатора и критерия
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Обучение
    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        device=device
    )
    
    # Сохраняем лучшую модель с явным именем
    torch.save(model.state_dict(), 'best_model.pth')
    
    print("\n✅ Модель обучена и сохранена как 'best_model.pth'")

if __name__ == '__main__':
    main()