from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15)
    dob = models.DateField()
    hospital_name = models.CharField(max_length=100)

class Patient(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    patient_id = models.CharField(max_length=50)
    weight = models.FloatField()
    temperature = models.FloatField()
    heart_rate = models.FloatField()
    symptoms = models.TextField()
    existing_condition = models.TextField()
    lab_test_result = models.TextField()
    cholesterol = models.FloatField()
    blood_sugar = models.FloatField()
    family_history = models.TextField()
    smoking = models.CharField(max_length=50)
    systolic_bp = models.FloatField()
    diastolic_bp = models.FloatField()
    bmi = models.FloatField()
    prediction_result = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Patient {self.patient_id} - {self.prediction_result}"