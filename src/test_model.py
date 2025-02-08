import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.cnn import SimpleCNN
from utils import load_config, set_seed, get_device
import matplotlib.pyplot as plt

def test_model(model_path, config_path='config/config.json'):
    # Загрузка конфигурации
    config = load_config(config_path)
    
    # Установка случайности
    set_seed(config['random_seed'])
    
    # Выбор устройства
    device = get_device(config)
    
    # Подготовка трансформаций
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Загрузка тестового набора
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False
    )
    
    # Создание модели
    model = SimpleCNN(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    # Загрузка весов
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Критерий
    criterion = nn.CrossEntropyLoss()
    
    # Тестирование
    print(f"🔍 Тестирование модели: {model_path}")
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Построение матрицы ошибок
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # Финальные метрики
    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total_samples
    
    print("\n📊 Результаты тестирования:")
    print(f"   Потеря: {test_loss:.4f}")
    print(f"   Точность: {test_accuracy:.2f}%")
    
    # Визуализация матрицы ошибок
    print("\n🔢 Матрица ошибок:")
    print(confusion_matrix)
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'confusion_matrix': confusion_matrix,
        'model': model,
        'test_loader': test_loader,
        'device': device
    }

def visualize_predictions(model, test_loader, device, num_images=5):
    model.eval()
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 1:  # Берем первый батч
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for j in range(min(num_images, images.size(0))):
                img = images[j].cpu().squeeze().numpy()
                axes[j].imshow(img, cmap='gray')
                title = f"Истина: {labels[j].item()}\nПредсказание: {predicted[j].item()}"
                axes[j].set_title(title)
                axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Тестирование нейронной сети')
    parser.add_argument('model_path', type=str, help='Путь к файлу с весами модели')
    parser.add_argument('--config', type=str, default='config/config.json', help='Путь к конфигурационному файлу')
    
    args = parser.parse_args()
    
    # Выполнить тестирование и получить модель
    results = test_model(args.model_path, args.config)
    
    # Использовать модель из результатов тестирования
    visualize_predictions(results['model'], results['test_loader'], results['device'])

if __name__ == '__main__':
    main() 