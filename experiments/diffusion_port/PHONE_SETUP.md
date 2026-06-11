# Запуск Mercury-порта на Android (OnePlus 15) — `qwen_mercury_phone.py`

Скрипт делает то же, что Kaggle-ноутбук (`B* = C(donor)`, B не обучается, данные — только
самогенерация донора), но целиком на CPU, с чекпойнтами/резюмом (Android убивает долгие
процессы) и телефонными бюджетами.

## Что реально влезает
| Модель | fp32 RAM | Вердикт |
|---|---|---|
| Qwen3.5-0.8B | ~3.2 GB весов, пик ~5–6 GB | **OK** на 12–16 GB телефоне (~1–1.5 ч) |
| supra50m | ~0.2 GB | летает (минуты) |
| Qwen3.5-9B | 36 GB fp32 / 18 GB fp16 | **не влезает ни в один телефон** — только GPU |

## Установка (Termux + proot Ubuntu)
1. Поставь **Termux с F-Droid** (версия из Play Store сломана).
2. В Termux:
   ```bash
   pkg update -y && pkg install -y proot-distro
   proot-distro install ubuntu
   proot-distro login ubuntu
   ```
3. В Ubuntu (aarch64-манивheels ставятся штатно):
   ```bash
   apt update && apt install -y python3 python3-pip git
   pip3 install torch --index-url https://download.pytorch.org/whl/cpu
   pip3 install transformers accelerate safetensors
   ```
4. Забери скрипт:
   ```bash
   git clone -b claude/kimi-k2-6-architecture-M5FaG https://github.com/gundoneg/mylittlepony
   cd mylittlepony/experiments/diffusion_port
   ```

## Запуск
```bash
# не давать Android усыпить процесс (выполнить в ОТДЕЛЬНОЙ termux-сессии, вне proot):
termux-wake-lock

# пилот по умолчанию (Qwen3.5-0.8B скачается с HF, ~1.6 GB):
python3 qwen_mercury_phone.py

# если процесс убили — продолжить с чекпойнта:
python3 qwen_mercury_phone.py --resume

# вдвое быстрее (без KD-учителя):
python3 qwen_mercury_phone.py --kd 0
```
Телефон на зарядку; закрой тяжёлые приложения (нужно ~6 GB свободной RAM). Прогресс
сохраняется в `mercury_phone/` каждые 50 шагов (`--save-every`).

## Что увидишь
Тот же протокол, что на Kaggle: отчёт об архитектуре (layer_types, инвентарь Linear),
floor-vs-B\* masked-CE по уровням шума, реконструкция, контроль перемешанных сигнатур,
semi-AR генерация, и на выходе `mercury_pack.safetensors` (перевод: UA/V-факторы на каждую
матрицу + [MASK]-эмбеддинг) + чекпойнт C.

## Проверено
Скрипт прогнан end-to-end (реальный transformers 5.11) против локального supra50m —
архитектурно-агностичный C нашёл 84 Linear-матрицы, обучение/evals/pack работают.
Для Qwen3.5: DeltaNet-conv на CPU идёт штатным fp32-фолбэком transformers (телефону не
нужны CUDA-кернелы — кстати, поэтому телефон обходит ровно ту проблему, что убила T4).

## Честные рамки
- 9B на телефоне невозможен физически — это пилотная платформа; 9B добиваем на GPU
  (Kaggle с фиксами v2 или Colab).
- DeltaNet-слои Qwen3.5 причинны по конструкции: bidirectional становятся только
  full-attention слои (1 из 4) — архитектурный потолок гибрида, не дефект метода.
