# Face Mask Detection (Telegram + FastAPI)

Пример клиент–серверного приложения на Python для классификации ношения маски по фото. Используется датасет `Face Mask Dataset` с классами `WithMask` и `WithoutMask`.

## Структура проекта
- `config.yaml` — пути и основные гиперпараметры.
- `ml_workflow/` — директория с ML-компонентами:
  - `train/` — скрипты обучения:
    - `train_hog_lr.py` — HOG + LogisticRegression.
    - `train_mobilenet_v2.py` — MobileNetV2 (transfer learning).
    - `train_resnet18_lr.py` — ResNet18 embeddings + LogisticRegression.
    - `utils.py` — датасеты, метрики, аугментации, HOG, детектор лиц.
  - `eval/evaluate_models.py` — оценка на тесте, метрики, ROC/CM.
  - `models/` — сохраняются веса и артефакты моделей.
- `server/app.py` — FastAPI с `/predict` (`model=hog_lr|mobilenet_v2|resnet18_lr`).
- `bot/telegram_bot.py` — Telegram-бот (polling), пересылает фото на сервер.

## Быстрый старт
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Обучение

**Классический пайплайн (HOG + LogisticRegression):**
```bash
python ml_workflow/train/train_hog_lr.py --apply-face-crop
```

**Глубокая модель (MobileNetV2, transfer learning):**
```bash
python ml_workflow/train/train_mobilenet_v2.py --epochs 20 --batch-size 32 --img-size 224
```

**Гибрид (ResNet18 embeddings + LogisticRegression):**
```bash
python ml_workflow/train/train_resnet18_lr.py --img-size 224 --batch-size 64
```

Логи и метрики сохраняются в `logs/` и `outputs/`, лучшие веса — в `ml_workflow/models/`.

## Оценка на тесте
```bash
python ml_workflow/eval/evaluate_models.py --model all
```
Скрипт сохранит метрики (`outputs/test_metrics.json`) и графики ROC/CM (`outputs/plots/`).

## Запуск сервера
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Запуск Telegram-бота
**Переменные окружения**:
  - `TELEGRAM_TOKEN=<токен бота>`
  - `SERVER_URL=http://localhost:8000/predict`

**Запуск:**
```bash
python bot/telegram_bot.py
```
