import socket
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import os

# ===== Configuration =====
HOST = "192.168.100.4"
PORT = 8084
LOG_PREFIX = "hand_sensor_data_"

# ===== Create folders =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOOD_DIR = os.path.join(BASE_DIR, "GOOD")
BAD_DIR = os.path.join(BASE_DIR, "BAD")

os.makedirs(GOOD_DIR, exist_ok=True)
os.makedirs(BAD_DIR, exist_ok=True)

# ===== Server Setup =====
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"[SERVER] Listening on {HOST}:{PORT} ...")
client_socket, addr = server.accept()
print(f"[CONNECTED] Device connected from {addr}")

# ===== Global Variables =====
recording = False
writer = None
file = None
current_type = "Word"
current_hand = "Right"
current_quality = "Good"

# ===== GUI Functions =====
def start_recording():
    global recording, writer, file, current_type, current_hand, current_quality
    if recording:
        return
    recording = True
    
    # تحديد المجلد حسب الجودة
    if current_quality == "Good":
        save_dir = GOOD_DIR
    else:
        save_dir = BAD_DIR
    
    # اسم الملف
    filename = os.path.join(
        save_dir,
        f"{LOG_PREFIX}{current_quality}_{current_type}_{current_hand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    
    file = open(filename, mode="w", newline="")
    writer = csv.writer(file)
    writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "type", "quality", "hand"])
    status_label.config(text=f"Recording... ({current_quality})", foreground="green")
    print(f"[INFO] Recording started: {filename}")

def stop_recording():
    global recording, file
    if not recording:
        return
    recording = False
    if file:
        file.close()
    status_label.config(text="Recording stopped.", foreground="red")
    print("[INFO] Recording stopped.")

def set_type(value):
    global current_type
    current_type = value

def set_hand(value):
    global current_hand
    current_hand = value

def set_quality(value):
    global current_quality
    current_quality = value

# ===== GUI Setup =====
root = tk.Tk()
root.title("Smart Pen Data Recorder")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

title_label = tk.Label(root, text="Arabic Handwriting Recorder", font=("Arial", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

# Hand selection
hand_frame = tk.LabelFrame(root, text="Select Hand", padx=10, pady=5, bg="#f0f0f0")
hand_frame.pack(pady=5, fill="x")
hand_var = tk.StringVar(value="Right")
tk.Radiobutton(hand_frame, text="Right", variable=hand_var, value="Right", command=lambda: set_hand("Right"), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
tk.Radiobutton(hand_frame, text="Left", variable=hand_var, value="Left", command=lambda: set_hand("Left"), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)

# Quality selection
quality_frame = tk.LabelFrame(root, text="Select Quality", padx=10, pady=5, bg="#f0f0f0")
quality_frame.pack(pady=5, fill="x")
quality_var = tk.StringVar(value="Good")
tk.Radiobutton(quality_frame, text="Good ", variable=quality_var, value="Good", command=lambda: set_quality("Good"), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
tk.Radiobutton(quality_frame, text="Bad ", variable=quality_var, value="Bad", command=lambda: set_quality("Bad"), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)

# Type selection
type_frame = tk.LabelFrame(root, text="Select Type", padx=10, pady=5, bg="#f0f0f0")
type_frame.pack(pady=5, fill="x")
type_var = tk.StringVar(value="Word")
tk.Radiobutton(type_frame, text="Word", variable=type_var, value="Word", command=lambda: set_type("Word"), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
tk.Radiobutton(type_frame, text="Sentence", variable=type_var, value="Sentence", command=lambda: set_type("Sentence"), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)

# Start/Stop buttons
btn_frame = tk.Frame(root, bg="#f0f0f0")
btn_frame.pack(pady=15)
tk.Button(btn_frame, text="Start Recording", bg="green", fg="white", font=("Arial", 12, "bold"), command=start_recording).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="Stop Recording", bg="red", fg="white", font=("Arial", 12, "bold"), command=stop_recording).pack(side=tk.LEFT, padx=10)

# Status label
status_label = tk.Label(root, text="Idle", font=("Arial", 12), bg="#f0f0f0")
status_label.pack(pady=10)

# ===== Data Reception Loop =====
def receive_data():
    global recording, writer, current_type, current_hand, current_quality
    if recording:
        try:
            data = client_socket.recv(1024).decode("utf-8")
            if data:
                for line in data.strip().splitlines():
                    fields = line.split(",")
                    if len(fields) == 7:
                        timestamp = fields[0]
                        ax, ay, az = fields[1], fields[2], fields[3]
                        gx, gy, gz = fields[4], fields[5], fields[6]

                        row = [timestamp, ax, ay, az, gx, gy, gz, current_type, current_quality, current_hand]
                        writer.writerow(row)
                        file.flush()
                        print(f"Received: {row}")
        except Exception as e:
            print("Error receiving:", e)
    root.after(100, receive_data)

root.after(100, receive_data)
root.mainloop()

# ===== Cleanup =====
client_socket.close()
server.close()
print("[SERVER] Shutdown complete.")