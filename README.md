# ObjectDetection-DRL
# 🔐 Real-Time DRL-Based Smart Surveillance System  
# 📹 Gerçek Zamanlı DRL Destekli Akıllı Güvenlik Sistemi

This project implements a smart surveillance system that detects people and dangerous objects (like knives and scissors) in real time using YOLOv5 and makes intelligent alarm decisions through Deep Reinforcement Learning (DQN).  
Bu proje, gerçek zamanlı kişi ve tehlikeli alet (bıçak, makas) tespiti yapan ve Derin Pekiştirmeli Öğrenme (DQN) ile akıllı alarm kararları veren bir güvenlik sistemi sunmaktadır.


## 🚀 Features / Özellikler

- 🔍 **YOLOv5-based Detection**: Detects people, knives, and scissors with bounding boxes.
- 🧠 **Deep Q-Learning Agent**: Learns when to trigger alarms based on presence of people and dangerous tools.
- 🔊 **Voice Alert (TTS)**: Speaks warning when a threat is detected.
- 🧮 **Person Counting**: Counts total and current people in the frame.
- 🎥 **Auto Video Recording**: Tracks and records person with dangerous object.
- 🧠 **Experience Replay + Training**: Learns optimal policy with SmoothL1 loss.
- 💾 **Model Saving**: Automatically saves trained model (`.pth` file).


## 🖥️ Technologies Used / Kullanılan Teknolojiler

- Python
- OpenCV
- PyTorch
- YOLOv5 (Ultralytics Hub)
- pyttsx3 (Text-to-Speech)
- Deep Q-Network (DQN)
- Google Colab / Local GPU / CPU


## 🎯 System Workflow / Sistem Akışı

1. Kamera açılır, çerçeve alınır.
2. YOLOv5 kişi ve tehlikeli aleti tespit eder.
3. DRL ajanı alarm verip vermemeye karar verir.
4. Tehdit algılanırsa:
   - Sesli uyarı (TTS)
   - Ekran görüntüsü kaydı
   - Video takibi başlar
5. Kişi sayısı güncellenir ve alarm kararları öğrenilir.
6. Eğitim sonrası model `.pth` olarak kaydedilir.


## 🧪 DRL State & Reward Logic

`python
# State: [person_detected, dangerous_tool_detected]
# Action: 0 = no alarm, 1 = trigger alarm
# Reward:
#   +1.0 → Alarm correct (person + weapon)
#   -1.0 → False alarm
#   -0.5 → Missed detection
#   +0.1 → Safe, no alarm

⚠️ Note: This is a research/demo system and not intended for real deployment without safety checks.
⚠️ Not: Bu sistem sadece akademik/deneysel amaçlıdır; gerçek kullanıma uygun hale getirilmeden kullanılmamalıdır.

Developer
İkra Sarıçiçek
Computer Engineering, Biruni University
📧 ikra.saricicek2@gmail.com
