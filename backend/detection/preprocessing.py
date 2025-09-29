import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from django.conf import settings

# MediaPipe
import mediapipe as mp
import cv2

from .model import STGCNModel   # your ST-GCN architecture

# -------------------
# Load ST-GCN Model
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_nodes = 75
num_classes = 52
A = torch.eye(num_nodes)

model = STGCNModel(in_channels=2, num_class=num_classes, A=A).to(device)
model.load_state_dict(torch.load(
    os.path.join(settings.BASE_DIR, "models", "best_stgcn_paksign.pth"),
    map_location=device
))
model.eval()

label_map = {
    0: "Afraid (خوفزدہ)", 1: "Allow (اجازت دینا)", 2: "Angry (غصہ)",
    3: "Appreciate (تعریف کرنا)", 4: "Ask (پوچھنا)", 5: "Autumn (خزاں)",
    6: "Baby (بچہ)", 7: "Bank (بینک)", 8: "Book (کتاب)", 9: "Broken (ٹوٹا ہوا)",
    10: "Brother (بھائی)", 11: "Calm (پرسکون)", 12: "Child (بچہ)", 13: "Children (بچے)",
    14: "Climb (چڑھنا)", 15: "Cloud (بادل)", 16: "Cold (سرد)", 17: "Comb (کنگھی)",
    18: "Come (آنا)", 19: "Complete (مکمل)", 20: "Computer (کمپیوٹر)", 21: "Daily (روزانہ)",
    22: "Difficult (مشکل)", 23: "Doctor (طبیب/ڈاکٹر)", 24: "Door (دروازہ)", 25: "Drink (پینا)",
    26: "Early (جلدی)", 27: "Eat (کھانا)", 28: "Fail (ناکام ہونا)", 29: "Fan (پنکھا)",
    30: "Father (والد)", 31: "Fight (لڑنا)", 32: "Future (مستقبل)", 33: "Happy (خوش)",
    34: "Home (گھر)", 35: "Knock (دستک دینا)", 36: "Nice (اچھا)", 37: "Night (رات)",
    38: "Open (کھولنا)", 39: "Order (حکم)", 40: "Pass (پاس کرنا)", 41: "Read (پڑھنا)",
    42: "Sad (اداس)", 43: "Say (کہنا)", 44: "Scissors (قینچی)", 45: "See (دیکھنا)",
    46: "Sleep (سونا)", 47: "Smile (مسکراہٹ)", 48: "Tomorrow (کل)", 49: "Turn off (بند کرنا)",
    50: "Turn on (چلانا)", 51: "You (تم)"
}

# -------------------
# Video → NPZ
# -------------------
def video_to_npz(video_path, output_path, num_joints=75):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            keypoints = []

            # Pose (33 joints)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    keypoints.append([lm.x, lm.y])
            else:
                keypoints.extend([[0, 0]] * 33)

            # Left hand (21 joints)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y])
            else:
                keypoints.extend([[0, 0]] * 21)

            # Right hand (21 joints)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    keypoints.append([lm.x, lm.y])
            else:
                keypoints.extend([[0, 0]] * 21)

            # Pad/truncate to num_joints
            if len(keypoints) < num_joints:
                keypoints.extend([[0, 0]] * (num_joints - len(keypoints)))
            elif len(keypoints) > num_joints:
                keypoints = keypoints[:num_joints]

            frames.append(keypoints)

    cap.release()
    sequence = np.array(frames)  # (T, V, 2)
    np.savez_compressed(output_path, sequence=sequence)
    return output_path


# -------------------
# Prediction
# -------------------
def predict_sign(npz_path):
    sample = np.load(npz_path)
    sequence = sample["sequence"]

    # Ensure shape (C, T, V)
    if sequence.ndim == 3:  # (T, V, C)
        T, V, C = sequence.shape
        sequence = sequence[:, :, :2]  # keep only x,y
        sequence = sequence.transpose(2, 0, 1)  # (C, T, V)
    else:
        raise ValueError(f"Unexpected sequence shape: {sequence.shape}")

    sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sequence)
        probs = F.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    return label_map[pred_class], confidence
