from django.db import models
from django.contrib.auth.models import AbstractUser


class CustomUser(AbstractUser):
    """Custom user model aligned with initial migration (extends AbstractUser)."""
    # No additional fields for now; matches the generated migration
    pass
