import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import torchaudio
import torch
from audiocraft.models import MusicGen
from deep_translator import GoogleTranslator

model = None
ui_q = queue.Queue()

def jp_to_en(text):
    return GoogleTranslator(source='ja', target='en').translate(text)

def load_model_worker():
    global model
    try:
        ui_q.put(("splash_status", "モデル読み込み中...\n（初回は時間がかかります）"))
        model = MusicGen.get_pretrained("facebook/musicgen-small")
        # 必要ならGPUへ: model.to('cuda')
        model.set_generation_params(duration=8)
        ui_q.put(("splash_status", "モデル読み込み完了！"))
    except Exception as e:
        ui_q.put(("error", f"モデル読み込みエラー: {e}"))
    finally:
        ui_q.put(("close_splash", None))

def generate_worker(prompt_jp, file_path, duration):
    try:
        ui_q.put(("status", "翻訳中..."))
        prompt_en = jp_to_en(prompt_jp)

        if model is None:
            ui_q.put(("error", "モデルがロードされていません"))
            return

        model.set_generation_params(duration=duration)
        ui_q.put(("status", f"音楽生成中...\n({prompt_en})"))

        with torch.no_grad():
            wavs = model.generate([prompt_en])
        wav = wavs[0].cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(file_path, wav, 32000)
        ui_q.put(("done", f"保存完了: {file_path}"))
    except Exception as e:
        ui_q.put(("error", str(e)))
    finally:
        ui_q.put(("stop_progress", None))

# --- GUI 作成 ---
root = tk.Tk()
root.title("Music")
root.geometry("420x250")
root.withdraw()  # ← 最初は隠す

# スプラッシュ中央表示用関数
def center_window(win, width, height):
    win.update_idletasks()
    screen_w = win.winfo_screenwidth()
    screen_h = win.winfo_screenheight()
    x = (screen_w // 2) - (width // 2)
    y = (screen_h // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")

# スプラッシュ
splash = tk.Toplevel()
splash.title("読み込み中")
center_window(splash, 300, 120)  # 中央表示
splash.resizable(False, False)
splash_label = tk.Label(splash, text="準備中...", font=("Arial", 12))
splash_label.pack(expand=True)

# メインウィジェット
tk.Label(root, text="日本語で音楽の説明を入力:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

duration_var = tk.IntVar(value=8)
tk.Label(root, text="生成秒数:").pack()
tk.Scale(root, from_=4, to=15, orient="horizontal", variable=duration_var).pack()

generate_button = tk.Button(root, text="音楽生成")
generate_button.pack(pady=10)

progress_bar = ttk.Progressbar(root, mode="indeterminate", length=300)
progress_bar.pack(pady=5)

status_label = tk.Label(root, text="モデル読み込み準備中...")
status_label.pack(pady=5)

# UI のメッセージ処理ループ
def process_ui_queue():
    try:
        while True:
            action, payload = ui_q.get_nowait()
            if action == "splash_status":
                splash_label.config(text=payload)
            elif action == "close_splash":
                try:
                    splash.destroy()
                except:
                    pass
                root.deiconify()
                center_window(root, 420, 250)  # メインも中央表示
            elif action == "status":
                status_label.config(text=payload)
            elif action == "stop_progress":
                progress_bar.stop()
                generate_button.config(state='normal')
            elif action == "done":
                status_label.config(text=payload)
                messagebox.showinfo("完了", "音楽生成が完了しました")
            elif action == "error":
                messagebox.showerror("エラー", payload)
                status_label.config(text="エラーが発生しました")
                generate_button.config(state='normal')
    except queue.Empty:
        pass
    root.after(100, process_ui_queue)

# ボタンコールバック
def on_generate_btn():
    if model is None:
        messagebox.showerror("エラー", "モデルがまだ読み込まれていません")
        return
    prompt_jp = entry.get().strip()
    if not prompt_jp:
        messagebox.showerror("エラー", "日本語プロンプトを入力してください")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAVファイル", "*.wav")])
    if not file_path:
        return
    generate_button.config(state='disabled')
    progress_bar.start(10)
    threading.Thread(target=generate_worker, args=(prompt_jp, file_path, duration_var.get()), daemon=True).start()

generate_button.config(command=on_generate_btn)

# モデル読み込みスレッド開始
threading.Thread(target=load_model_worker, daemon=True).start()
process_ui_queue()
root.mainloop()