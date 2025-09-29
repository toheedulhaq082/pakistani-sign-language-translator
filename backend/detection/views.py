import os
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .preprocessing import video_to_npz, predict_sign

@api_view(["POST"])
def predict_video(request):
    try:
        file_obj = request.FILES["file"]

        # Save uploaded video
        video_path = os.path.join(settings.MEDIA_ROOT, file_obj.name)
        with open(video_path, "wb+") as dest:
            for chunk in file_obj.chunks():
                dest.write(chunk)

        # Convert to NPZ
        npz_path = video_path.replace(".mp4", ".npz")
        video_to_npz(video_path, npz_path)

        # Predict
        sign, conf = predict_sign(npz_path)

        return Response({
            "prediction": sign,
            "confidence": round(conf, 3)
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
