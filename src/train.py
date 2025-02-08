import torch
import torch.nn as nn
import torch.optim as optim
import sys

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Простой прогресс-бар без внешних библиотек
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    sys.stdout.flush()
    if iteration == total:
        print()

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience=5, device='cpu'):
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\n🚀 Эпоха {epoch+1}/{epochs}")
        
        # Обучение
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Прогресс-бар
            print_progress_bar(
                batch_idx + 1, 
                len(train_loader), 
                prefix='Обучение:', 
                suffix=f'Потеря: {loss.item():.4f}, Точность: {100 * correct / total_samples:.2f}%'
            )
        
        # Валидация
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
                # Прогресс-бар валидации
                print_progress_bar(
                    batch_idx + 1, 
                    len(val_loader), 
                    prefix='Валидация:', 
                    suffix=f'Потеря: {val_loss / (batch_idx + 1):.4f}, Точность: {100 * correct_val / total_val_samples:.2f}%'
                )
        
        # Метрики эпохи
        train_accuracy = 100 * correct / total_samples
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct_val / total_val_samples
        
        print(f"📊 Метрики эпохи:")
        print(f"   Обучение - Потеря: {total_loss / len(train_loader):.4f}, Точность: {train_accuracy:.2f}%")
        print(f"   Валидация - Потеря: {val_loss:.4f}, Точность: {val_accuracy:.2f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("💾 Сохранена лучшая модель")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"🛑 Остановка обучения. Валидационная потеря не улучшается {patience} эпох.")
            break
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('best_model.pth'))
    return model