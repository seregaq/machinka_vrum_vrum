import tkinter as tk
from tkinter import ttk
import threading
from tkinter import filedialog
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torchaudio
from scipy.io import wavfile
import tempfile
import os
from datetime import datetime



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




MODEL_PATH = "asr_model_control_acc_0.7000_20251222_1710.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

alphabet = checkpoint["alphabet"]
vocab = checkpoint["vocab"]
char2idx = checkpoint["char2idx"]
idx2char = checkpoint["idx2char"]

model = CTCModel(
    n_mels=checkpoint["mel_transform"]["n_mels"],
    n_classes=len(vocab)
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Модель загружена.")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=checkpoint["mel_transform"]["sample_rate"],
    n_mels=checkpoint["mel_transform"]["n_mels"],
    n_fft=checkpoint["mel_transform"]["n_fft"],
    hop_length=checkpoint["mel_transform"]["hop_length"]
)
mel_transform.eval()

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

def preprocess_audio(wav, sr=16000):
    
    wav_tensor = torch.tensor(wav, dtype=torch.float32)
    wav_tensor = (wav_tensor - wav_tensor.mean()) / (wav_tensor.std() + 1e-7)

    return wav_tensor


def extract_features(wav_tensor, sr=16000):
    if sr != 16000:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
    
    feats = mel_transform(wav_tensor.unsqueeze(0))
    feats = torch.log(feats + 1e-9)
    feats = feats.squeeze(0).transpose(0, 1)
    
    return feats


def save_recorded_audio(audio_data, sr=16000):
    temp_dir = tempfile.mkdtemp(prefix="asr_recordings_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(temp_dir, f"recording_{timestamp}.wav")
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(filename, sr, audio_int16)
    
    print(f"Аудио сохранено: {filename}")
    return filename


def load_and_process_audio_file(filename):
    try:

        sr, audio = wavfile.read(filename)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            pass  
        else:
            audio = audio.astype(np.float32)
        
        return audio, sr
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return None, None


def recognize_from_file(filename):

    audio, sr = load_and_process_audio_file(filename)
    if audio is None:
        return "Ошибка загрузки файла"
    

    wav_tensor = preprocess_audio(audio, sr)
    feats = extract_features(wav_tensor, sr)
    feats = feats.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(feats)
        logits = logits.log_softmax(dim=-1)
    
    feat_lens = torch.tensor([logits.size(1)], device=DEVICE)
    text = decode(logits, feat_lens)[0]
    
    return text



class ASRApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Demка репа")
        self.geometry("600x400")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")
        

        self.recordings_dir = "live_recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)

        # Заголовок
        title_label = ttk.Label(
            self,
            text="Распознавание слов",
            font=("Arial", 16, "bold"),
            background="#f0f0f0"
        )
        title_label.pack(pady=15)

        self.status_label = ttk.Label(
            self,
            text="✅ Модель загружена. Готов к записи.",
            wraplength=550,
            justify="center",
            background="#f0f0f0"
        )
        self.status_label.pack(pady=10)
        
        self.recording_indicator = tk.Label(
            self,
            text="",
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="red"
        )
        self.recording_indicator.pack()

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        
        self.record_button = tk.Button(
            button_frame,
            text="🎙 ЗАПИСЬ",
            command=self.start_recording,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=20,
            pady=10,
            relief="raised",
            borderwidth=3,
            width=10
        )
        self.record_button.pack(side=tk.LEFT, padx=5)
        
        self.file_button = tk.Button(
            button_frame,
            text="📂 ЗАГРУЗИТЬ WAV",
            command=self.load_and_recognize_file,
            bg="#2196F3",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=20,
            pady=10,
            relief="raised",
            borderwidth=3,
            width=12
        )
        self.file_button.pack(side=tk.LEFT, padx=5)

        result_frame = ttk.Frame(self)
        result_frame.pack(pady=20)
        
        ttk.Label(result_frame, text="Распознано:", 
                 font=("Arial", 12), background="#f0f0f0").pack()
        
        self.result_label = tk.Label(
            result_frame,
            text="—",
            font=("Arial", 24, "bold"),
            bg="#e8f5e9",
            fg="#2e7d32",
            width=20,
            height=2,
            relief="solid",
            borderwidth=1,
            wraplength=400
        )
        self.result_label.pack(pady=5)
        
        self.file_info_label = ttk.Label(
            self,
            text="",
            font=("Arial", 9),
            foreground="#666",
            background="#f0f0f0"
        )
        self.file_info_label.pack(pady=5)

        info_label = ttk.Label(
            self,
            text="Произнесите одно из слов: дом, окно, стол, дверь, свет...\nЗаписывается в файл → загружается → распознается",
            wraplength=550,
            justify="center",
            font=("Arial", 10),
            foreground="#666",
            background="#f0f0f0"
        )
        info_label.pack(pady=10)

        self.recording = False
        self.recording_file = None

    def start_recording(self):
        if self.recording:
            return
            
        self.recording = True
        self.record_button.config(state="disabled", bg="#cccccc")
        self.file_button.config(state="disabled", bg="#cccccc")
        self.result_label.config(text="—")
        self.file_info_label.config(text="")
        self.status_label.config(text="🎤 Записываю... Говорите сейчас!")
        self.recording_indicator.config(text="● ЗАПИСЬ")

        threading.Thread(target=self.record_and_recognize, daemon=True).start()

    def record_and_recognize(self):
        duration = 2.0  
        sr = 16000
        
        try:

            audio = sd.rec(
                int(duration * sr),
                samplerate=sr,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            
            audio = audio.squeeze()
            
            if np.max(np.abs(audio)) < 0.01:
                self.after(0, self.update_ui, "", "⚠️ Слишком тихо. Попробуйте еще раз.", "")
                return
            
            self.after(0, self.update_ui, "", "💾 Сохраняю файл...", "")
            
      
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_file = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
            
      
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(self.recording_file, sr, audio_int16)
            
            self.after(0, self.update_ui, "", "🤖 Распознаю из файла...", self.recording_file)
            
            text = recognize_from_file(self.recording_file)
            
            if not text or text == "(не распознано)":
                text = "(не распознано)"
                self.after(0, self.update_ui, text, "❌ Не удалось распознать. Попробуйте еще раз.", self.recording_file)
            else:
                self.after(0, self.update_ui, text, "✅ Распознано из файла!", self.recording_file)
                
        except Exception as e:
            self.after(0, self.update_ui, "Ошибка", f"❌ Ошибка: {str(e)}", "")
            
        finally:
            self.recording = False
            self.after(0, self.enable_buttons)

    def load_and_recognize_file(self):
        
        filename = filedialog.askopenfilename(
            title="Выберите WAV файл",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        self.record_button.config(state="disabled")
        self.file_button.config(state="disabled")
        self.result_label.config(text="—")
        self.status_label.config(text="🤖 Распознаю выбранный файл...")
        self.file_info_label.config(text=f"Файл: {os.path.basename(filename)}")
        
        threading.Thread(target=self.recognize_file_thread, args=(filename,), daemon=True).start()
    
    def recognize_file_thread(self, filename):
        try:
            text = recognize_from_file(filename)
            if not text or text == "(не распознано)":
                text = "(не распознано)"
                self.after(0, self.update_ui, text, "❌ Не удалось распознать файл.", filename)
            else:
                self.after(0, self.update_ui, text, "✅ Файл распознан!", filename)
        except Exception as e:
            self.after(0, self.update_ui, "Ошибка", f"❌ Ошибка: {str(e)}", filename)
        finally:
            self.after(0, self.enable_buttons)

    def update_ui(self, text, status, file_info=""):
        self.result_label.config(text=text)
        self.status_label.config(text=status)
        self.recording_indicator.config(text="")
        
        if file_info:
            if isinstance(file_info, str) and os.path.exists(file_info):
                file_size = os.path.getsize(file_info) / 1024
                self.file_info_label.config(
                    text=f"Файл: {os.path.basename(file_info)} ({file_size:.1f} KB)"
                )
            else:
                self.file_info_label.config(text=file_info)

    def enable_buttons(self):
        self.record_button.config(state="normal", bg="#4CAF50")
        self.file_button.config(state="normal", bg="#2196F3")



if __name__ == "__main__":

    try:
        print("Текущее устройство ввода:")
        print(sd.query_devices(sd.default.device[0], "input"))
    except:
        print("Не удалось получить информацию об устройстве ввода")
        
    app = ASRApp()
    app.mainloop()