import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from datetime import datetime
from collections import deque
import pyttsx3
import traceback

# === TTS Motoru Başlatma ===
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"TTS motoru başlatılamadı: {e}. Sesli uyarılar devre dışı.")
    engine = None

# === YOLOv5 Modeli Yükleme ===
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    CLASSES_DATA = model.names


    def find_class_id_flexible(class_name_to_find, model_classes_data):
        if isinstance(model_classes_data, dict):
            for class_id, name in model_classes_data.items():
                if name == class_name_to_find: return int(class_id)
            return -1
        elif isinstance(model_classes_data, list):
            try:
                return model_classes_data.index(class_name_to_find)
            except ValueError:
                return -1
        return -1


    PERSON_CLASS_ID = find_class_id_flexible('person', CLASSES_DATA)
    KNIFE_CLASS_ID = find_class_id_flexible('knife', CLASSES_DATA)
    SCISSORS_CLASS_ID = find_class_id_flexible('scissors', CLASSES_DATA)

    if PERSON_CLASS_ID == -1: PERSON_CLASS_ID = 0
    if KNIFE_CLASS_ID == -1: KNIFE_CLASS_ID = 43 # COCO'da knife
    if SCISSORS_CLASS_ID == -1: SCISSORS_CLASS_ID = 77 # COCO'da scissors (yolov5m+ gibi modellerde)

    DANGEROUS_TOOL_IDS = []
    if KNIFE_CLASS_ID != -1: DANGEROUS_TOOL_IDS.append(KNIFE_CLASS_ID)
    if SCISSORS_CLASS_ID != -1: DANGEROUS_TOOL_IDS.append(SCISSORS_CLASS_ID)
    if not DANGEROUS_TOOL_IDS:
        print("UYARI: Modelde tanımlı tehlikeli alet ID'si bulunamadı.")

    if isinstance(CLASSES_DATA, dict):
        max_id = -1
        if CLASSES_DATA and all(isinstance(k, int) for k in CLASSES_DATA.keys()): max_id = max(CLASSES_DATA.keys())
        CLASSES_list_from_dict = [""] * (max_id + 1) if max_id != -1 else []
        if max_id != -1:
            for k, v in CLASSES_DATA.items():
                if isinstance(k, int) and 0 <= k <= max_id: CLASSES_list_from_dict[k] = v
        CLASSES = CLASSES_list_from_dict
    elif isinstance(CLASSES_DATA, list):
        CLASSES = CLASSES_DATA
    else:
        CLASSES = []
    print(f"YOLOv5 modeli yüklendi. Person ID={PERSON_CLASS_ID}, Tehlikeli Alet ID'leri={DANGEROUS_TOOL_IDS}")

except Exception as e:
    print(f"YOLOv5 modeli yüklenirken hata oluştu: {e}");
    traceback.print_exc();
    exit()


# === Ortam Bilgisi ve Ödül Fonksiyonu ===
def get_drl_state(detections_np):
    person_detected = any(int(cls) == PERSON_CLASS_ID for *_, cls in detections_np)
    dangerous_tool_detected = any(int(cls) in DANGEROUS_TOOL_IDS for *_, cls in detections_np)
    return np.array([int(person_detected), int(dangerous_tool_detected)], dtype=np.float32)


def calculate_reward(current_state_arr, executed_action):
    person, dangerous_tool = current_state_arr
    if executed_action == 1: # Alarm ver
        return 1.0 if person and dangerous_tool else -1.0
    else: # Alarm verme
        return -0.5 if person and dangerous_tool else 0.1


# === Q-Network ===
class DeepQNetwork(nn.Module):
    def __init__(self, input_features=2, num_actions=2):
        super(DeepQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, num_actions))

    def forward(self, state_input): return self.layers(state_input)


# === Ajan ve DRL Parametreleri ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drl_policy_net = DeepQNetwork().to(device)
optimizer = optim.AdamW(drl_policy_net.parameters(), lr=1e-4, amsgrad=True)
criterion = nn.SmoothL1Loss()
REPLAY_MEMORY_CAPACITY = 10000
replay_memory = deque(maxlen=REPLAY_MEMORY_CAPACITY)
GAMMA_FACTOR = 0.99
EPS_START_EXPLORATION = 1.0
EPS_END_EXPLORATION = 0.05
current_epsilon = EPS_START_EXPLORATION
EPS_DECAY_RATE_PER_STEP = 0.9995
MINI_BATCH_SIZE = 128
previous_drl_state = None
previous_drl_action = None


def choose_action(state_np_array):
    global current_epsilon
    if random.random() < current_epsilon: return random.randrange(2) # 0: alarm verme, 1: alarm ver
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_np_array).unsqueeze(0).to(device)
        return int(torch.argmax(drl_policy_net(state_tensor)).item())


def optimize_reinforcement_learning_model():
    if len(replay_memory) < MINI_BATCH_SIZE: return
    transitions_batch = random.sample(replay_memory, MINI_BATCH_SIZE)
    state_batch_np, action_batch_np, reward_batch_np, next_state_batch_np = zip(*transitions_batch)
    state_batch_tensor = torch.FloatTensor(np.array(state_batch_np)).to(device)
    action_batch_tensor = torch.LongTensor(action_batch_np).unsqueeze(1).to(device)
    reward_batch_tensor = torch.FloatTensor(reward_batch_np).to(device)
    next_state_batch_tensor = torch.FloatTensor(np.array(next_state_batch_np)).to(device)
    q_values_for_actions_taken = drl_policy_net(state_batch_tensor).gather(1, action_batch_tensor).squeeze()
    with torch.no_grad():
        next_state_q_values = drl_policy_net(next_state_batch_tensor).max(1)[0]
        expected_q_values_for_actions = reward_batch_tensor + (GAMMA_FACTOR * next_state_q_values)
    loss = criterion(q_values_for_actions_taken, expected_q_values_for_actions)
    optimizer.zero_grad();
    loss.backward()
    torch.nn.utils.clip_grad_norm_(drl_policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    global current_epsilon
    if current_epsilon > EPS_END_EXPLORATION:
        current_epsilon = max(EPS_END_EXPLORATION, current_epsilon * EPS_DECAY_RATE_PER_STEP)


# === Kamera Ayarları ===
CAMERA_INDEX = 0
video_capture = cv2.VideoCapture(CAMERA_INDEX)
if not video_capture.isOpened(): print(f"Hata: Kamera {CAMERA_INDEX} açılamadı."); exit()
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === Genel Değişkenler ===
drl_triggered_alarm_on = False # İlk DRL alarmının (resim kaydetme, TTS) çalışması için False olmalı
session_people_entered_count = 0
persons_in_previous_frame = 0

# === Video Kayıt Değişkenleri (TAKİP EDEREK KAYIT) ===
RECORDING_FPS = 20.0
RECORDING_TIME_SECONDS = 30
MAX_FRAMES_FOR_RECORDING = int(RECORDING_FPS * RECORDING_TIME_SECONDS)

is_currently_recording = False
video_file_writer = None
fixed_recording_crop_width = 0
fixed_recording_crop_height = 0
current_target_person_bbox_for_tracking = None
frames_recorded_count = 0


def find_person_to_track(all_current_detections, tool_ids):
    person_detected_in_frame = False
    tool_detected_in_frame = False
    first_person_bbox = None

    for *box, conf, cls_id_float in all_current_detections:
        class_id = int(cls_id_float)
        if class_id == PERSON_CLASS_ID:
            person_detected_in_frame = True
            if first_person_bbox is None:
                first_person_bbox = tuple(map(int, box))
        elif class_id in tool_ids:
            tool_detected_in_frame = True

    if person_detected_in_frame and tool_detected_in_frame:
        return first_person_bbox
    return None


print("DRL Destekli Güvenlik Kamera Sistemi Başlatılıyor...")
try:
    while True:
        is_frame_read, current_frame = video_capture.read()
        if not is_frame_read: print("Kare okunamadı."); break

        frame_for_yolo_detection = current_frame.copy()
        yolo_detection_results = model(frame_for_yolo_detection)
        detections_from_yolo_np = yolo_detection_results.pred[0].cpu().numpy()

        drl_current_state = get_drl_state(detections_from_yolo_np)
        action_chosen_by_drl = choose_action(drl_current_state)

        persons_in_current_frame_count = 0
        target_person_for_video = find_person_to_track(detections_from_yolo_np, DANGEROUS_TOOL_IDS)

        for *object_bounding_box, confidence_score, class_id_float in detections_from_yolo_np:
            class_id = int(class_id_float)
            x1_obj, y1_obj, x2_obj, y2_obj = map(int, object_bounding_box)
            obj_bbox_tuple = (x1_obj, y1_obj, x2_obj, y2_obj)
            label_str = f"{CLASSES[class_id] if CLASSES and 0 <= class_id < len(CLASSES) and CLASSES[class_id] else f'ID:{class_id}'} {float(confidence_score):.2f}"

            is_dangerous_tool = class_id in DANGEROUS_TOOL_IDS
            is_target_person = target_person_for_video and obj_bbox_tuple == target_person_for_video

            display_color = (0, 0, 255) if is_dangerous_tool or is_target_person else (0, 255, 0)
            thickness = 3 if is_dangerous_tool or is_target_person else 2

            cv2.rectangle(current_frame, (x1_obj, y1_obj), (x2_obj, y2_obj), display_color, thickness)
            cv2.putText(current_frame, label_str, (x1_obj, y1_obj - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)

            if class_id == PERSON_CLASS_ID: persons_in_current_frame_count += 1
            if is_target_person:
                cv2.putText(current_frame, "TAKİP EDİLEN (TEHLİKE!)", (x1_obj, y1_obj - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if persons_in_current_frame_count > persons_in_previous_frame:
            session_people_entered_count += (persons_in_current_frame_count - persons_in_previous_frame)
        persons_in_previous_frame = persons_in_current_frame_count

        if action_chosen_by_drl == 1 and drl_current_state[0] == 1 and drl_current_state[1] == 1:
            cv2.putText(current_frame, "DRL ALARM: Kisi ve Tehlikeli Alet!", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)
            if not drl_triggered_alarm_on: # Sadece bir kez tetiklenir (alarm durumu devam ederken tekrar etmez)
                timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"uyari_drl_{timestamp_now}.jpg", current_frame)
                print(f"DRL Alarmı: uyari_drl_{timestamp_now}.jpg kaydedildi.")
                if engine:
                    try:
                        engine.say("DRL Uyarısı! Kişi ve tehlikeli alet tespit edildi!");
                        engine.runAndWait()
                    except Exception as e_tts:
                        print(f"Sesli uyarı hatası: {e_tts}")
                drl_triggered_alarm_on = True # Alarm verildi olarak işaretle
        else:
            drl_triggered_alarm_on = False # Alarm durumu bittiğinde, bir sonraki için hazırla

        should_record_this_frame = drl_current_state[0] == 1 and drl_current_state[1] == 1 and DANGEROUS_TOOL_IDS

        if should_record_this_frame and not is_currently_recording:
            current_target_person_bbox_for_tracking = target_person_for_video
            if current_target_person_bbox_for_tracking:
                fixed_recording_crop_width = current_target_person_bbox_for_tracking[2] - \
                                             current_target_person_bbox_for_tracking[0]
                fixed_recording_crop_height = current_target_person_bbox_for_tracking[3] - \
                                              current_target_person_bbox_for_tracking[1]

                if not (fixed_recording_crop_width > 20 and fixed_recording_crop_height > 20):
                    fixed_recording_crop_width = 300
                    fixed_recording_crop_height = 300

                timestamp_now_rec = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_output_filename = f"kayit_kisi_ve_alet_{timestamp_now_rec}.mp4"
                video_codec = cv2.VideoWriter_fourcc(*'mp4v')

                video_file_writer = cv2.VideoWriter(video_output_filename, video_codec, RECORDING_FPS,
                                                    (fixed_recording_crop_width, fixed_recording_crop_height))
                is_currently_recording = True
                frames_recorded_count = 0 # Her yeni kayıt başladığında kare sayacını sıfırla
                print(
                    f"Video kaydı (Kişi+Alet) başlatıldı: {video_output_filename} Kırpma: {fixed_recording_crop_width}x{fixed_recording_crop_height}")
            else:
                print("Kayıt başlatılamadı: Takip edilecek kişi bulunamadı (drl_state ile tutarsızlık).")

        if is_currently_recording and video_file_writer is not None:
            # Kayıt devam koşulu: Hala hem kişi hem de alet var mı (drl_current_state)?
            if (drl_current_state[1] == 1 and drl_current_state[1]==0) or (drl_current_state[0] == 1 ):
                if target_person_for_video: # Her karede hedef kişiyi güncelle
                    current_target_person_bbox_for_tracking = target_person_for_video

                if current_target_person_bbox_for_tracking:
                    p_x1, p_y1, p_x2, p_y2 = current_target_person_bbox_for_tracking
                    center_x = (p_x1 + p_x2) // 2
                    center_y = (p_y1 + p_y2) // 2

                    crop_x1 = center_x - fixed_recording_crop_width // 2
                    crop_y1 = center_y - fixed_recording_crop_height // 2
                    crop_x2 = crop_x1 + fixed_recording_crop_width
                    crop_y2 = crop_y1 + fixed_recording_crop_height

                    frame_h_live, frame_w_live, _ = current_frame.shape
                    crop_x1_c = max(0, crop_x1)
                    crop_y1_c = max(0, crop_y1)
                    crop_x2_c = min(frame_w_live, crop_x2)
                    crop_y2_c = min(frame_h_live, crop_y2)

                    if crop_x1_c < crop_x2_c and crop_y1_c < crop_y2_c:
                        sub_frame_for_video = current_frame[crop_y1_c:crop_y2_c, crop_x1_c:crop_x2_c]
                        if sub_frame_for_video.size > 0:
                            resized_sub_frame_for_video = cv2.resize(sub_frame_for_video, (
                            fixed_recording_crop_width, fixed_recording_crop_height))
                            video_file_writer.write(resized_sub_frame_for_video)
                            cv2.rectangle(current_frame, (crop_x1_c, crop_y1_c), (crop_x2_c, crop_y2_c), (0, 165, 255), 2)
                            rec_text = f"KAYIT ({frames_recorded_count})" # Sadece kare sayısını göster
                            text_y = crop_y1_c - 10 if crop_y1_c > 20 else crop_y1_c + 20
                            cv2.putText(current_frame, rec_text, (crop_x1_c, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 165, 255), 2)
                    frames_recorded_count += 1
                else:
                    # Takip edilecek kişi bulunamadı, ancak drl_state hala kişi+alet diyor.
                    # Bu durumda tüm frame'i kaydetmek yerine, bbox'u olmayan kareyi atlayabiliriz.
                    # Veya son bilinen bbox etrafında kayda devam edilebilir (mevcut mantık bunu yapar)
                    # Eğer current_target_person_bbox_for_tracking None ise, ve kayıt devam ediyorsa
                    # bu blok aslında çalışmaz, çünkü yukarıdaki if current_target_person_bbox_for_tracking:
                    # false olur. Eğer kayıt devam ederken kişi bir anlığına kaybolursa ve
                    # current_target_person_bbox_for_tracking güncellenmezse, son bilinen bölgeyi kaydeder.
                    # Eğer find_person_to_track hiçbir zaman bir şey bulamazsa ve kayıt başlarsa,
                    # o zaman bu 'else' nadiren çalışır. Genelde 'target_person_for_video' dolu olur.
                    print("Kayıt devam ediyor ancak bu karede takip edilecek bbox yok.")
            else:  # Kişi veya alet kayboldu -> Kaydı durdur
                if is_currently_recording: # Sadece kayıt gerçekten devam ediyorsa durdurma işlemleri yap
                    is_currently_recording = False
                    current_target_person_bbox_for_tracking = None # Takip hedefini temizle
                    if video_file_writer: video_file_writer.release()
                    video_file_writer = None
                    status_msg = "kişi ve/veya alet kayboldu"
                    print(f"Video kaydı durduruldu ({status_msg}). {frames_recorded_count} kare yazıldı.")
                    # frames_recorded_count bir sonraki kayıt için zaten başlangıçta sıfırlanıyor.

        if previous_drl_state is not None and previous_drl_action is not None:
            reward = calculate_reward(previous_drl_state, previous_drl_action)
            replay_memory.append(
                (previous_drl_state, previous_drl_action, reward, drl_current_state))
        if len(replay_memory) > MINI_BATCH_SIZE: optimize_reinforcement_learning_model()
        previous_drl_state = drl_current_state
        previous_drl_action = action_chosen_by_drl

        info_y_start = current_frame.shape[0] - 110
        cv2.putText(current_frame, f"Toplam Giren: {session_people_entered_count}", (10, info_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(current_frame, f"Anlık Kişi: {persons_in_current_frame_count}", (10, info_y_start + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(current_frame, f"Epsilon: {current_epsilon:.4f}", (10, info_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (100, 200, 200), 2)
        cv2.putText(current_frame, f"Bellek: {len(replay_memory)}/{REPLAY_MEMORY_CAPACITY}", (10, info_y_start + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 200), 2)

        cv2.imshow("DRL Destekli Kamera", current_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Çıkış yapılıyor..."); break
        elif key == ord('s'):
            path = f"drl_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            torch.save(drl_policy_net.state_dict(), path);
            print(f"Model kaydedildi: {path}")

except Exception as e_main:
    print(f"Ana döngüde hata: {e_main}");
    traceback.print_exc()
finally:
    print("Temizlik yapılıyor...");
    video_capture.release();
    cv2.destroyAllWindows()
    if video_file_writer is not None: print("Kalan video kaydı kapatılıyor..."); video_file_writer.release()
    try:
        torch.save(drl_policy_net.state_dict(), "drl_final.pth"); print("Son model kaydedildi.")
    except Exception as e_save:
        print(f"Son model kaydedilemedi: {e_save}")
    print("Program sonlandı.")