import os
import json
import cv2
import numpy as np
import torch
import mediapipe as mp
import tempfile
from tqdm import tqdm
import traceback

# Import the necessary model and graph classes from the repository's source
# NOTE: You must ensure 'src.model' and 'src.dataset.graphs' are accessible in your Django environment
# I'll assume they are available via PYTHONPATH or by placing this file appropriately.
from src.model import create as create_model
from src.dataset.graphs import Graph


# Define a directory for saving the debug videos
DEBUG_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "sign_debug_chunks") 
# Get the temporary directory path
if not os.path.exists(DEBUG_OUTPUT_DIR):
    os.makedirs(DEBUG_OUTPUT_DIR)

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic



# --- CONFIGURATION CONSTANTS (MATCHING predict.py) ---
NUM_FRAMES = 64
NUM_JOINTS = 65 
NUM_CHANNELS = 2
DATA_SHAPE = [2, NUM_CHANNELS, NUM_FRAMES, NUM_JOINTS, 1] 

# 65-Joint Index mapping (Based on your predict.py's implementation)
# Indices 23 to 43 are Left Hand (21 joints)
# Indices 44 to 64 are Right Hand (21 joints)
LEFT_HAND_WRIST_POSE_IDX = 15 
RIGHT_HAND_WRIST_POSE_IDX = 16
# We'll use the pose wrists (15 and 16) as they are less noisy than hand base landmarks

# --- BONE CONNECTIONS (Copied from predict.py) ---
def remap_index(original_index):
    """Remaps a 75-joint index to the new 65-joint scheme."""
    if 0 <= original_index <= 22: return original_index
    elif 33 <= original_index <= 53: return original_index - 10 
    elif 54 <= original_index <= 74: return original_index - 10
    else: return -1

BASE_BONE_CONNECTIONS_75 = [
    # POSE CONNECTIONS (Simplified upper body)
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21), # Left Arm
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), # Right Arm
    
    # HAND CONNECTIONS 
    (33, 34), (34, 35), (35, 36), (36, 37), (33, 38), (38, 39), (39, 40), (40, 41), 
    (33, 42), (42, 43), (43, 44), (44, 45), (33, 46), (46, 47), (47, 48), (48, 49), 
    (33, 50), (50, 51), (51, 52), (52, 53), 
    (54, 55), (55, 56), (56, 57), (57, 58), (54, 59), (59, 60), (60, 61), (61, 62), 
    (54, 63), (63, 64), (64, 65), (65, 66), (54, 67), (67, 68), (68, 69), (69, 70), 
    (54, 71), (71, 72), (72, 73), (73, 74) 
]

BONE_CONNECTIONS = []
for start_idx_75, end_idx_75 in BASE_BONE_CONNECTIONS_75:
    start_idx_65 = remap_index(start_idx_75)
    end_idx_65 = remap_index(end_idx_75)
    if start_idx_65 != -1 and end_idx_65 != -1:
        BONE_CONNECTIONS.append((start_idx_65, end_idx_65))

NUM_BONES = len(BONE_CONNECTIONS)

# --- Helper Functions (Copied/Modified from predict.py) ---

def extract_keypoints(results_landmarks):
    """Helper function to extract keypoints (x, y, z) from MediaPipe results."""
    keypoints = []
    if results_landmarks:
        for lm in results_landmarks.landmark:
            # We only extract X, Y (the first NUM_CHANNELS)
            keypoints.extend([lm.x, lm.y]) # We only need 2D data for prediction
    return np.array(keypoints, dtype=np.float32)

def calculate_bone_features(joint_data_2d):
    """Calculates 2D bone displacement vectors for the 61 bones."""
    T, V, C = joint_data_2d.shape
    bone_data = np.zeros((T, NUM_BONES, C))

    for i, (start_idx, end_idx) in enumerate(BONE_CONNECTIONS):
        bone_vector = joint_data_2d[:, end_idx, :] - joint_data_2d[:, start_idx, :]
        bone_data[:, i, :] = bone_vector

    return bone_data


# --- Core Preprocessing Functions ---
def extract_full_video_keypoints(video_path):
    """
    Processes the entire video to extract all 65 2D keypoints per frame.
    Returns: A tuple (joint_data_2d, raw_frames, frame_rate)
             joint_data_2d: np.array of shape (Total_Frames, 65, 2)
             raw_frames: list of cv2 image arrays
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames_2d = []
    raw_frames = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Use a simpler Holistic model for extraction if possible, but stick to the original for consistency
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5) as holistic:
        for _ in tqdm(range(total_frames), desc="Extracting full video keypoints"):
            ret, frame = cap.read()
            if not ret: break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make a copy of the frame before processing for saving later
            raw_frames.append(frame.copy()) 
            
            results = holistic.process(image_rgb)

            # --- Keypoint Extraction Logic (Same as before) ---
            pose_kps_33_2d = extract_keypoints(results.pose_landmarks).reshape(-1, NUM_CHANNELS)
            left_hand_kps_2d = extract_keypoints(results.left_hand_landmarks).reshape(-1, NUM_CHANNELS)
            right_hand_kps_2d = extract_keypoints(results.right_hand_landmarks).reshape(-1, NUM_CHANNELS)
            
            # Padding for missing detections
            if pose_kps_33_2d.shape[0] == 0: pose_kps_33_2d = np.zeros((33, NUM_CHANNELS))
            if left_hand_kps_2d.shape[0] == 0: left_hand_kps_2d = np.zeros((21, NUM_CHANNELS))
            if right_hand_kps_2d.shape[0] == 0: right_hand_kps_2d = np.zeros((21, NUM_CHANNELS))
            
            pose_kps_65_2d = pose_kps_33_2d[:23, :] 
            all_kps_65_2d = np.vstack([pose_kps_65_2d, left_hand_kps_2d, right_hand_kps_2d])
            frames_2d.append(all_kps_65_2d)
    
    cap.release()
    
    if not frames_2d:
        raise ValueError("Could not extract any keypoints from the video.")

    return np.array(frames_2d), raw_frames, frame_rate


def segment_signs(joint_data_2d, raw_frames, frame_rate, min_gap_sec=.25, movement_threshold=0.005):
    """
    Segments the full keypoint sequence into individual sign chunks and saves
    the resulting video chunk files with skeleton overlay for visual inspection.
    Returns: A list of keypoint sequences (the data chunks for the model).
    """
    T, V, C = joint_data_2d.shape
    
    # Get the Y-coordinates of the left and right wrist/pose (indices 15 and 16)
    left_wrist_y = joint_data_2d[:, LEFT_HAND_WRIST_POSE_IDX, 1]
    right_wrist_y = joint_data_2d[:, RIGHT_HAND_WRIST_POSE_IDX, 1]
    hand_y = np.minimum(left_wrist_y, right_wrist_y)
    velocity = np.diff(hand_y, prepend=hand_y[0])

    min_gap_frames = int(min_gap_sec * frame_rate)
    
    sign_chunks_data = []
    in_sign = False
    sign_start_frame = 0
    chunk_count = 0
    
    H, W, _ = raw_frames[0].shape if raw_frames else (0, 0, 0)
    
    print(f"\nSaving debug video chunks to: {DEBUG_OUTPUT_DIR}")

    for i in range(1, T):
        # Condition 1: Sign START (hands move up)
        if velocity[i] < -movement_threshold and not in_sign:
            sign_start_frame = max(0, i - 1)
            in_sign = True
            
        # Condition 2: Sign END (hands move down, then a period of stillness)
        elif velocity[i] > movement_threshold and in_sign:
            # Check for a subsequent period of low velocity (rest)
            rest_end = min(T, i + min_gap_frames)
            is_still = np.max(np.abs(velocity[i:rest_end])) < movement_threshold
            
            if is_still:
                sign_end_frame = i
                
                # Check for minimum sign length
                if sign_end_frame > sign_start_frame + 5: # Min 5 frames of activity
                    chunk_kps = joint_data_2d[sign_start_frame:sign_end_frame]
                    chunk_frames = raw_frames[sign_start_frame:sign_end_frame]
                    
                    sign_chunks_data.append(chunk_kps)
                    chunk_count += 1
                    
                    # --- VISUALIZATION AND SAVING ---
                    
                    # Setup video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    debug_filename = os.path.join(DEBUG_OUTPUT_DIR, f"sign_chunk_{chunk_count}.mp4")
                    out = cv2.VideoWriter(debug_filename, fourcc, frame_rate, (W, H))

                    # Draw skeleton on each frame and write to video
                    for frame in chunk_frames:
                        annotated_frame = frame.copy()
                        
                        # Convert 65-joint normalized KPs back to MediaPipe format for drawing (complex)
                        # Instead, let's re-run MediaPipe on the raw frame for easy drawing
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = mp_holistic.Holistic(static_image_mode=True).process(image_rgb)

                        # Draw pose landmarks (using simplified connections)
                        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        # Draw hands
                        mp_drawing.draw_landmarks(annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        
                        # Mark segmentation frames with text
                        cv2.putText(annotated_frame, f'Sign {chunk_count}', (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        
                        out.write(annotated_frame)

                    out.release()
                    print(f"-> Saved: {debug_filename}")

                in_sign = False
                
    # Handle the final segment if the video ends mid-sign
    if in_sign and T > sign_start_frame + 5:
        chunk_kps = joint_data_2d[sign_start_frame:T]
        chunk_frames = raw_frames[sign_start_frame:T]
        
        sign_chunks_data.append(chunk_kps)
        chunk_count += 1
        
        # NOTE: Implement final frame saving logic here if desired, or skip for brevity
        
    print(f"\n--- Segmentation Complete: {chunk_count} sign chunks processed. ---")
    return sign_chunks_data



def prepare_chunk_tensor(joint_data_2d_chunk):
    """
    Takes a single sign's keypoint chunk (T, 65, 2) and prepares it into 
    the final 6D multi-stream tensor (1, I, C, T, V, M).
    This includes sampling/padding to NUM_FRAMES=64, bone feature calculation, and normalization.
    """
    current_frames = joint_data_2d_chunk.shape[0]
    
    # 1. Temporal Sampling/Padding
    if current_frames > NUM_FRAMES:
        indices = np.linspace(0, current_frames - 1, NUM_FRAMES, dtype=int)
        joint_data_2d = joint_data_2d_chunk[indices]
    elif current_frames < NUM_FRAMES:
        padding = np.zeros((NUM_FRAMES - current_frames, NUM_JOINTS, NUM_CHANNELS))
        joint_data_2d = np.concatenate([joint_data_2d_chunk, padding])
    else:
        joint_data_2d = joint_data_2d_chunk
    
    # 2. Normalization for Joints (Data Stream)
    data_j = (joint_data_2d - 0.5) * 2.0
    
    # 3. Bone Features (Bone Stream)
    data_b_unnorm = calculate_bone_features(joint_data_2d)
    data_b = (data_b_unnorm - 0.5) * 2.0 
    
    # 4. Reshape and Pad BONE stream to match JOINTS (V=65)
    # Final Shape: (C, T, V, 1)
    data_j = data_j.transpose(2, 0, 1) # (T, V, C) -> (C, T, V)
    data_j = np.expand_dims(data_j, axis=-1) # (C, T, V, 1)

    data_b = data_b.transpose(2, 0, 1) # (C, T, B)
    data_b = np.expand_dims(data_b, axis=-1) # (C, T, B, 1)

    padding_needed = NUM_JOINTS - NUM_BONES
    padding_shape = (NUM_CHANNELS, NUM_FRAMES, padding_needed, 1)
    padding = np.zeros(padding_shape, dtype=data_b.dtype)
    data_b_padded = np.concatenate([data_b, padding], axis=2) # (C, T, 65, 1)
    
    # 5. Final Multi-Stream Stack (I, C, T, V, M)
    multi_stream_tensor = np.stack([data_j, data_b_padded], axis=0) # (2, C, T, V, 1)
    
    # Add batch dimension: (1, I, C, T, V, M)
    return torch.from_numpy(multi_stream_tensor).float().unsqueeze(0)


# --- Model Loading and Prediction ---

# Model path and label map loading should be done ONCE when Django starts
# We'll make load_model a standalone function to be called on server start (or once per view call for simplicity)

def load_model_for_inference(model_path, num_classes=20):
    """
    Loads the trained EfficientGCN-B0 model and its weights from a checkpoint file.
    (Copied and modified from predict.py)
    """
    print(f"Loading model from: {model_path}")

    # Use 'cuda' if available, otherwise 'cpu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        graph = Graph(dataset='psl_mediapipe') 
        A_tensor = torch.from_numpy(graph.A).float()
    except Exception as e:
        print(f"Error loading Graph class: {e}. Check PYTHONPATH for 'src/dataset/graphs'.")
        # Return a dummy graph if graph loading fails, but this will fail later.
        raise e
    
    # --- MODEL ARGUMENTS MATCHING OPTIMIZED B0 TRAINING ---
    model_args = {
        'stem_channel': 64, 
        'block_args': [[16, 1, 0], [16, 1, 0], [32, 2, 1], [64, 2, 1]], 
        'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG',
        'drop_prob': 0.4, 
        'kernel_size': [5,2], 'scale_args': [1.2,1.35],
        'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True,
        'in_channels': 2, 'data_shape': DATA_SHAPE, 'A': A_tensor,
        'num_class': num_classes,
    }
    
    model = create_model('EfficientGCN-B0', **model_args)

    try:
        # Load checkpoint weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model checkpoint not found. Please check the path: {model_path}")
        
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}.")
    return model, device

def predict_signs(keypoint_sequences, model, device, label_map):
    """
    Processes all segmented sign chunks and returns a sentence.
    """
    predicted_signs = []
    
    if not keypoint_sequences:
        return []

    with torch.no_grad():
        for i, chunk in enumerate(tqdm(keypoint_sequences, desc="Predicting signs")):
            try:
                # 1. Prepare tensor (1, I, C, T, V, M)
                video_tensor_6d = prepare_chunk_tensor(chunk).to(device)
                
                # 2. Model Inference
                outputs, _ = model(video_tensor_6d)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 3. Get best prediction
                # torch.max returns (values, indices)
                confidence_tensor, top1_index_tensor = torch.max(probabilities, dim=1) 
                
                top1_index = top1_index_tensor.item()
                confidence = confidence_tensor.item() # <-- Call .item() on the tensor, not the tuple
                
                sign_name = label_map.get(top1_index, "UNKNOWN_SIGN")
                
                print(f"Chunk {i+1}: Predicted Sign: {sign_name} ({confidence:.2%})")
                predicted_signs.append(sign_name)
                
            except Exception as e:
                print(f"Error predicting chunk {i}: {e}")
                predicted_signs.append("ERROR_PREDICTING")
                
    return predicted_signs