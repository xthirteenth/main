1) Start main.py: дожидаемся полного конца обучения/прерывания
2) Запускаем скрипт тестирования:

# Базовый запуск
python src/test_model.py best_model.pth

# С указанием кастомного конфига
python src/test_model.py best_model.pth --config my_config.json

3) Смотрим predictions.png