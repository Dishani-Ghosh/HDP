from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Patient

class RegisterForm(UserCreationForm):
    email=forms.EmailField(required=True)
    phone_number=forms.CharField(max_length=15, required=True)
    dob=forms.DateField(required=True, widget=forms.SelectDateWidget(years=range(1900,2024)))
    hospital_name = forms.CharField(required=True,max_length=100)



    class Meta:
        model = User
        fields = ['username','email','phone_number','dob','hospital_name','password1','password2']


class PatientForm(forms.ModelForm):
    
    class Meta:
        model = Patient
        fields = ['name', 'age', 'gender', 'symptoms']


class PredictionForm(forms.Form):
    patient_id = forms.CharField(max_length=20)
    weight = forms.FloatField()
    temperature = forms.FloatField()
    heart_rate = forms.IntegerField()
    symptoms = forms.CharField(widget=forms.Textarea)
    existing_conditions = forms.CharField(widget=forms.Textarea, required=False)
    cholesterol = forms.FloatField()
    blood_sugar = forms.FloatField()
    family_history = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    smoking_status = forms.ChoiceField(choices=[('Smoker', 'Smoker'), ('Non-Smoker', 'Non-Smoker')])
    systolic_bp = forms.IntegerField()
    diastolic_bp = forms.IntegerField()
    bmi = forms.FloatField()

