## 📌 Implementation Status

- [ ] `In Progress` [Hands-On] GPT-OSS 바닥부터 구현하기 - 정상근 교수님

---

## 이 저장소의 목적

**`Project/`에 있는 논문 저자 코드를 읽거나 유튜브의 다양한 Implementation 영상을 참고하여, `Scratch/`에서 최소 구조로 직접 구현해 보는** 연습용 저장소입니다.  
기본은 **PyTorch Lightning**이며, **Plain PyTorch** 구조도 씁니다.

```
~/Dev/Scratch/
├── README.md          ← 이 문서 (구조 가이드)
├── Project/           ← 📖 참고용: 저자들이 공개한 원본/공식 코드 (fork)
│   └── (모델 이름)/
└── Scratch/           ← ✍️ 내 구현: 아래 최소 구조로 직접 작성
    └── (모델 이름)
```

| 폴더 | 역할 | 건드리는 방식 |
|------|------|---------------|
| `Project/` | 논문 저자 코드 — **읽고 이해** | fork 그대로 두고 참고만 |
| `Scratch/` | 내가 **직접 구현** | 아래 최소 구조로 새로 작성 |

저자 코드를 통째로 복사하지 않고,  
핵심 아이디어만 이해한 뒤 **내 손으로 최소 구조**에 맞춰 다시 짭니다.

---

## 내 구현 (`Scratch/`) — Plain PyTorch 최소 구조

논문마다 `Scratch/<모델 이름>/` 아래에 아래 뼈대만 만듭니다.  
**epoch 루프, loss, optimizer를 직접 작성**할 때 사용합니다.

```
Scratch/<모델 이름>/       # 예시
├── configs/default.yaml        # 설정 (batch_size, lr, epoch 등)
├── data/
│   └── dataset.py              # Dataset (+ DataLoader 생성 헬퍼)
├── models/
│   └── model.py                # nn.Module — 구조만 (forward)
├── train.py                    # 학습 루프 (for epoch, loss, backward)
├── scripts/
│   └── inference.py            # checkpoint로 추론
├── utils.py                    # load_config, checkpoint 저장/로드 등
├── requirements.txt
└── .gitignore
```

### 각 폴더 역할 (Plain PyTorch)

| 폴더/파일 | 역할 |
|-----------|------|
| `configs/` | 실험 설정. yaml만 바꿔 실험 |
| `data/dataset.py` | `Dataset` + `DataLoader` 만드는 코드 |
| `models/model.py` | 신경망 **구조만** (`nn.Module`, `forward`) |
| `train.py` | config 로드 → model/data 조립 → **학습 loop 직접 작성** |
| `scripts/inference.py` | 학습된 checkpoint로 결과 생성 |
| `utils.py` | config, checkpoint, 이미지 저장 등 **공통 함수** |
| `requirements.txt` | `torch`, `diffusers` 등 pip 목록 |
| `.gitignore` | checkpoint, data 등 Git 제외 |

### `train.py`가 하는 일 (전부)

```python
# 1. yaml 로드
# 2. models/model.py → nn.Module 생성
# 3. data/dataset.py → DataLoader 생성
# 4. optimizer, loss 정의
# 5. for epoch → for batch → forward → loss.backward() → step
# 6. checkpoint 저장
```

`engine/` 폴더 없음 — 학습 로직이 **`train.py` 한 파일**에 들어갑니다.

---

## 내 구현 (`Scratch/`) — Lightning 최소 구조

Plain PyTorch로 익힌 뒤, **반복 코드(epoch, checkpoint, GPU)를 Trainer에 맡길 때** 이 구조를 씁니다.

논문마다 `Scratch/<모델 이름>/` 아래에 아래 뼈대만 만듭니다.

```
Scratch/<모델 이름>/       # 예시
├── configs/default.yaml        # 설정 (batch_size, lr, epoch, gpus)
├── data/
│   └── datamodule.py           # LightningDataModule
├── models/
│   └── model.py                # nn.Module — 구조만 (forward)
├── engine/
│   └── module.py               # LightningModule (training_step, optimizer)
├── main.py                     # Trainer.fit() — 실행 진입점
├── scripts/
│   └── inference.py            # (선택) checkpoint로 추론
├── requirements.txt
└── .gitignore
```

### 각 폴더 역할 (Lightning)

| 폴더/파일 | 역할 |
|-----------|------|
| `configs/` | 실험 설정. 코드 수정 없이 yaml만 바꿔 실험 |
| `data/` | Dataset + `LightningDataModule` (전처리, DataLoader) |
| `models/` | 신경망 **구조만** (`nn.Module`, `forward`) |
| `engine/` | **학습 로직** (`LightningModule`: loss, training_step) |
| `main.py` | config + model + data + `Trainer` 조립 후 `fit()` |
| `scripts/inference.py` | 학습된 checkpoint로 결과 생성 (Lightning 밖에서도 OK) |
| `requirements.txt` | `pytorch-lightning` 등 pip 목록 |
| `.gitignore` | checkpoint, data 등 Git 제외 |

### Plain PyTorch vs Lightning

| | Plain PyTorch | Lightning |
|---|---------------|-----------|
| 학습 loop | `train.py`에 직접 | `engine/module.py` + `Trainer.fit()` |
| DataLoader | `data/dataset.py` | `data/datamodule.py` |
| 진입점 | `train.py` | `main.py` |
| 공통 함수 | `utils.py` | `utils.py` (선택) |
| 언제 | PyTorch·학습 loop 공부 | multi-GPU, logging, checkpoint 필요할 때 |

### `main.py`가 하는 일 (Lightning)

```python
# 1. yaml 로드
# 2. engine/module.py  → LightningModule 생성
# 3. data/datamodule.py → LightningDataModule 생성
# 4. pl.Trainer(epoch, gpu, checkpoint)
# 5. trainer.fit(model, datamodule)
```

epoch 루프, `backward()`, checkpoint 저장은 **Trainer**가 처리합니다.

---


