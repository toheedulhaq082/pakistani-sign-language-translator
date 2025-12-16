import os
import sys
import json
import tempfile
import torch
import arabic_reshaper
from bidi.algorithm import get_display
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

MODEL_PROJECT_ROOT = r"D:\PakSign_Pose_EGCN\EfficientGCNv1"

if MODEL_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, MODEL_PROJECT_ROOT)

from . import preprocessing

# --- CONFIGURATION CONSTANTS ---
MODEL_PATH = r"D:\PakSign_Pose_EGCN\EfficientGCNv1\workdir\EfficientGCN-B0_92.93%.tar" 

# Updated Label Map with Urdu Translations
LABEL_MAP = {
    0: "پوچھو",      # Ask
    1: "غصہ",        # Anger
    2: "اجازت",      # Permission/Allow
    3: "آؤ",          # Come
    4: "کمپیوٹر",    # Computer
    5: "پیو",         # Drink
    6: "کھاؤ",        # Eat
    7: "والد",       # Father
    8: "قینچی",      # Scissors
    9: "گھر",         # Home
    10: "دستک",      # Knock
    11: "کھولو",      # Open
    12: "سو جاؤ",     # Sleep
    13: "کہو",        # Say
    14: "خوش",        # Happy
    15: "دیکھو",      # Look/See
    16: "اداس",     # Sad
    17: "آپ",       # You
    18: "بند کر دو",  # Turn off/Close
    19: "مسکراہٹ"    # Smile
}

MODEL = None
DEVICE = None

def load_global_model():
    """Loads the model once when the Django process starts."""
    global MODEL, DEVICE
    if MODEL is None:
        try:
            MODEL, DEVICE = preprocessing.load_model_for_inference(
                model_path=MODEL_PATH, 
                num_classes=len(LABEL_MAP)
            )
        except Exception as e:
            print(f"FATAL ERROR: Failed to load model! {e}")
            MODEL = None
            DEVICE = None

load_global_model() 

# --- HELPER FUNCTION FOR URDU TEXT ---
def format_urdu_text(text):
    """
    Reshapes Urdu text to join letters correctly and fixes direction (RTL).
    Useful for console logs or drawing text on images (OpenCV/PIL).
    """
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# -------------------------------------------------------------------

@api_view(["POST"])
def predict_video(request):
    if MODEL is None or DEVICE is None:
        return Response({"error": "Prediction model is not initialized."}, 
                        status=status.HTTP_503_SERVICE_UNAVAILABLE)

    if 'file' not in request.FILES:
        return Response({"error": "No video file found (expected key 'file')."}, 
                        status=status.HTTP_400_BAD_REQUEST)

    video_file = request.FILES['file']
    temp_video_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            temp_video_path = tmp.name

        # Extraction and Segmentation logic
        full_kps, raw_frames, frame_rate = preprocessing.extract_full_video_keypoints(temp_video_path)
        sign_chunks = preprocessing.segment_signs(full_kps, raw_frames, frame_rate)
        
        if not sign_chunks:
            return Response({
                "prediction": "No significant movement.",
                "prediction_display": "No significant movement.",
                "signs_detected": 0
            }, status=status.HTTP_200_OK)

        # Predict signs (This returns Urdu words now)
        predicted_signs = preprocessing.predict_signs(sign_chunks, MODEL, DEVICE, LABEL_MAP)
        
        # 1. Standard Unicode Sentence (Best for Flutter/Frontend)
        # Flutter handles Urdu rendering natively, so it needs the raw string.
        predicted_sentence = " ".join(predicted_signs)
        
        # 2. Reshaped Sentence (Best for Console/OpenCV/Debugging)
        # This fixes the "dismantled" letters in environments that don't support Urdu.
        predicted_sentence_display = format_urdu_text(predicted_sentence)

        # Print to server console to verify it looks correct there
        print(f"Predicted (Raw): {predicted_sentence}")
        print(f"Predicted (Reshaped): {predicted_sentence_display}")
        
        return Response({
            "prediction": predicted_sentence,          # Send this to Flutter
            "prediction_display": predicted_sentence_display, # Use this if printing to a non-RTL terminal
            "predictions_list": predicted_signs, 
            "signs_detected": len(predicted_signs),
            "debug_output_path": preprocessing.DEBUG_OUTPUT_DIR
        }, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"Error: {e}")
        return Response({"error": f"Processing Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)