import soundfile as sf
import torch.nn as nn
import torch
import torchaudio
import os
import csv
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import datetime
import warnings
import numpy as np

warnings.filterwarnings('ignore')
alphabet = list("абвгдежзийклмнопрстуфхцчшщыэюяь")
vocab = ["<blank>"] + alphabet
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}

WORDS = [
    "дом", "окно", "стол", "дверь", "свет", "старт", "стоп", 
    "вверх", "вниз", "левый", "правый", "один", "два", "три", "четыре",
    "пять", "шесть", "семь", "восемь", "девять"
]

class ASRCSVDataset(Dataset):
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"]
                text = row["text"]

                if not os.path.exists(path):
                    print(f"Предупреждение: файл не найден - {path}")
                    continue

                self.samples.append((path, text))

        print(f"Загружено {len(self.samples)} примеров")
        
        if len(self.samples) == 0:
            raise ValueError("ОШИБКА: Нет данных для обучения!")
            
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=400,
            hop_length=160
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]

        try:
            wav, sr = sf.read(path)
        except Exception as e:
            raise RuntimeError(f"Ошибка при чтении файла {path}: {e}")
        
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        
        wav = torch.tensor(wav, dtype=torch.float32)
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        feats = self.mel(wav.unsqueeze(0))
        feats = torch.log(feats + 1e-9)
        feats = feats.squeeze(0).transpose(0, 1)

        target = torch.tensor(
            [char2idx[c] for c in text],
            dtype=torch.long
        )

        return feats, target, text


def collate_fn(batch):
    feats, targets, words = zip(*batch)

    feat_lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    targ_lens = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    feats = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    targets = torch.cat(targets)

    return feats, targets, feat_lens, targ_lens, words


class CTCModel(nn.Module):
    def __init__(self, n_mels, n_classes,
                 hidden_size=256,
                 num_layers=3,
                 bidirectional=True,
                 dropout=0.3):
        super().__init__()

        self.bn = nn.BatchNorm1d(n_mels)
        self.embedding = nn.Linear(n_mels, hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        
        self.fc1 = nn.Linear(out_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        
        x = self.embedding(x)
        x, _ = self.lstm(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def decode(logits, feat_lens=None):
    preds = logits.argmax(dim=-1).cpu().numpy()
    if feat_lens is not None:
        feat_lens = feat_lens.cpu().numpy()

    results = []

    for i, seq in enumerate(preds):
        if feat_lens is not None:
            seq = seq[:feat_lens[i]]

        prev = -1
        word = []
        for p in seq:
            if p != 0 and p != prev:
                word.append(idx2char[p])
            prev = p

        results.append("".join(word))

    return results


def test_on_control_files(model, control_dir="dat_control", device="cpu"):

    print(f"\n{'='*60}")
    print("ТЕСТИРОВАНИЕ НА КОНТРОЛЬНЫХ ФАЙЛАХ")
    print(f"{'='*60}")
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160
    )
    
    control_files = []
    for word in WORDS:
        for f in os.listdir(control_dir):
            if f.lower().endswith('.wav'):
                file_word = f.split('_')[0].lower()
                if file_word == word:
                    control_files.append((os.path.join(control_dir, f), word))
    

    control_files = control_files[:10]
    if len(control_files) == 0:
        print("❌ В папке dat_control не найдено контрольных файлов!")
        return 0.0
    
    print(f"Найдено {len(control_files)} контрольных файлов:")
    for path, word in control_files:
        print(f"  - {Path(path).name} -> '{word}'")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for audio_path, true_word in control_files:
            try:
                wav, sr = sf.read(audio_path)
                
                if len(wav.shape) > 1:
                    wav = wav.mean(axis=1)
                
                wav = torch.tensor(wav, dtype=torch.float32)
                wav = (wav - wav.mean()) / (wav.std() + 1e-7)

                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)

                # Извлекаем признаки
                feats = mel_transform(wav.unsqueeze(0))
                feats = torch.log(feats + 1e-9)
                feats = feats.squeeze(0).transpose(0, 1)
                feats = feats.unsqueeze(0).to(device)
                
                # Распознаем
                logits = model(feats)
                preds = decode(logits)
                predicted_word = preds[0] if preds else ""
                
                # Сравниваем
                is_correct = (predicted_word == true_word)
                status = "✓" if is_correct else "✗"
                
                print(f"  {Path(audio_path).name:20} -> '{predicted_word:10}' (ожидалось: '{true_word}') {status}")
                
                if is_correct:
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"  ❌ Ошибка при обработке {audio_path}: {e}")
                total += 1  # Считаем как ошибку
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n📊 Результат: {correct}/{total} = {accuracy:.4f}")
    
    return accuracy



def main():

    batch_size = 32  
    epochs = 60
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    control_dir = "dat_control"
    csv_path = "./labels.csv"
    dataset = ASRCSVDataset(csv_path)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )



    model = CTCModel(80, len(vocab)).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\n{'='*60}")
    print("НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for features, targets, feat_lens, target_lens, words in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            feat_lens = feat_lens.to(device)
            target_lens = target_lens.to(device)

            optimizer.zero_grad()

            logits = model(features)
            logits = logits.log_softmax(dim=-1)
            logits_ctc = logits.permute(1, 0, 2)

            loss = criterion(logits_ctc, targets, feat_lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
        
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, targets, feat_lens, target_lens, words in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                feat_lens = feat_lens.to(device)
                target_lens = target_lens.to(device)

                logits = model(features)
                logits = logits.log_softmax(dim=-1)
                
                preds = decode(logits, feat_lens)
                
                targets_np = targets.cpu().numpy()
                target_lens_np = target_lens.cpu().numpy()
                
                start_idx = 0
                for length in target_lens_np:
                    seq = targets_np[start_idx:start_idx + length]
                    true_word = "".join([idx2char[c] for c in seq])
                    
                    if preds and len(preds) > val_total:
                        if preds[val_total] == true_word:
                            val_correct += 1
                    
                    val_total += 1
                    start_idx += length
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}")

    #контрольные данные
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    control_accuracy = test_on_control_files(model, control_dir, device)

    if control_accuracy >= 0.6:
        model_info = {
            'model_state_dict': model.state_dict(),
            'alphabet': alphabet,
            'vocab': vocab,
            'char2idx': char2idx,
            'idx2char': idx2char,
            'mel_transform': {
                'sample_rate': 16000,
                'n_mels': 80,
                'n_fft': 400,
                'hop_length': 160
            },
            'training_info': {
                'control_accuracy': control_accuracy,
                'epochs_trained': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'total_params': total_params,
                'test_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        filename = f"asr_model_control_acc_{control_accuracy:.4f}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pth"
        torch.save(model_info, filename)
        torch.save(model_info, "best_asr_model.pth")

        print(f"\n✅ Модель сохранена (контрольная точность: {control_accuracy:.4f}):")
        print(f"   - {filename}")
        print(f"   - best_asr_model.pth")
    else:
        print(f"\n❌ Контрольная точность {control_accuracy:.4f} < 0.75 — модель НЕ сохранена")

if __name__ == "__main__":
    main()