# Face Mask Detection (Telegram + FastAPI)

Полный пример клиент–серверного приложения на Python для классификации ношения маски по фото. Используется датасет `Face Mask Dataset` с классами `WithMask` и `WithoutMask`.

## Структура проекта
- `config.yaml` — пути и основные гиперпараметры.
- `train/` — скрипты обучения:
  - `train_hog_lr.py` — HOG + LogisticRegression.
  - `train_mobilenet_v2.py` — MobileNetV2 (transfer learning).
  - `train_resnet18_lr.py` — ResNet18 embeddings + LogisticRegression.
  - `utils.py` — датасеты, метрики, аугментации, HOG, детектор лиц.
- `eval/evaluate_models.py` — оценка на тесте, метрики, ROC/CM.
- `server/app.py` — FastAPI с `/predict` (`model=hog_lr|mobilenet_v2|resnet18_lr`).
- `bot/telegram_bot.py` — Telegram-бот (polling), пересылает фото на сервер.
- `models/` — сохраняются веса и артефакты моделей.
- `tests/` — базовые тесты (детектор лица, endpoint схемы).

## Быстрый старт
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Убедитесь, что датасет лежит в `Face Mask Dataset/` (как в репозитории).

## Обучение
Параметры можно менять через CLI или `config.yaml`.

Классический пайплайн (HOG + LogisticRegression):
```bash
python train/train_hog_lr.py --apply-face-crop
```

Глубокая модель (MobileNetV2, transfer learning):
```bash
python train/train_mobilenet_v2.py --epochs 20 --batch-size 32 --img-size 224
```

Гибрид (ResNet18 embeddings + LogisticRegression):
```bash
python train/train_resnet18_lr.py --img-size 224 --batch-size 64
```

Логи и метрики сохраняются в `logs/` и `outputs/`, лучшие веса — в `models/`.

## Оценка на тесте
```bash
python eval/evaluate_models.py --model all
```
Скрипт сохранит метрики (`outputs/test_metrics.json`) и графики ROC/CM (`outputs/plots/`).

## Запуск сервера
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Запуск Telegram-бота
1. Создайте `.env` или экспортируйте переменные:
   - `TELEGRAM_TOKEN=<токен бота>`
   - `SERVER_URL=http://localhost:8000/predict`
2. Запустите:
```bash
python bot/telegram_bot.py
```
Команды: `/start`, `/help`, `/model mobilenet_v2|hog_lr|resnet18_lr`. На фото бот отвечает результатами классификации с вероятностями.
