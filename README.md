# Face Mask Detection (Telegram + FastAPI)

Полный пример клиент–серверного приложения на Python для классификации ношения маски по фото. Используется датасет `Face Mask Dataset/` (подпапки `Train/`, `Validation/`, `Test/` с классами `WithMask` и `WithoutMask`).

## Структура проекта
- `config.yaml` — пути и основные гиперпараметры.
- `notebooks/01_data_exploration.ipynb` — EDA: баланс классов, повреждения, размеры, детекция лиц, выбросы.
- `train/` — скрипты обучения:
  - `train_classical.py` — HOG + LogisticRegression.
  - `train_dl.py` — MobileNetV2 (transfer learning).
  - `train_third.py` — ResNet18 embeddings + LogisticRegression.
  - `utils.py` — датасеты, метрики, аугментации, HOG, детектор лиц.
- `eval/evaluate_models.py` — оценка на тесте, метрики, ROC/CM.
- `server/app.py` — FastAPI с `/predict` (`model=classical|dl|third`).
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
python train/train_classical.py --apply-face-crop
```

Глубокая модель (MobileNetV2, transfer learning):
```bash
python train/train_dl.py --epochs 20 --batch-size 32 --img-size 224
```

Гибрид (ResNet18 embeddings + LogisticRegression):
```bash
python train/train_third.py --img-size 224 --batch-size 64
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
Эндпоинт:
```
POST /predict?model=dl
Form-Data: image=<файл>
```
Ответ:
```json
{
  "prediction": "masked",
  "probability": 0.92,
  "model": "dl",
  "details": { "face_detected": true, "face_bbox": [x, y, w, h] }
}
```

## Запуск Telegram-бота
1. Создайте `.env` или экспортируйте переменные:
   - `TELEGRAM_TOKEN=<токен бота>`
   - `SERVER_URL=http://localhost:8000/predict`
2. Запустите:
```bash
python bot/telegram_bot.py
```
Команды: `/start`, `/help`, `/model dl|classical|third`. На фото бот отвечает: «Человек в маске ✅ (confidence: 92%)».

## Тесты
```bash
pytest -q
```
Тесты используют флаг `USE_DUMMY_MODELS=1`, поэтому тяжелые веса не требуются.

## Рекомендации
- При дисбалансе классов используйте веса (`class_weight`) и аугментации (включены в DL-скрипт).
- Если детектор лиц часто ошибается, оставляйте fallback на полный кадр (реализовано).
- Для репродюсируемости фиксированы seed, сохранены логи и конфиги.

## Возможные расширения
- Экспорт модели в ONNX (добавить в `train/train_dl.py`).
- Dockerfile для сервера и бота.
- MLflow/W&B трекинг.

