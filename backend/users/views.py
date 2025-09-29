from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth import authenticate, logout, get_user_model

from .serializers import SignUpSerializer, SignInSerializer

User = get_user_model()


@api_view(["POST"])
def signup_view(request):
    serializer = SignUpSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        return Response(
            {
                "id": user.id,
                "email": user.email,
                "username": user.username
            },
            status=status.HTTP_201_CREATED
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def signin_view(request):
    serializer = SignInSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data["user"]
        return Response(
            {
                "id": user.id,
                "email": user.email,
                "username": user.username
            },
            status=status.HTTP_200_OK
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def logout_view(request):
    logout(request)
    return Response(
        {"message": "Logged out successfully"}, status=status.HTTP_200_OK
    )
