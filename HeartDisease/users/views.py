from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from .models import Patient, UserProfile
from .forms import RegisterForm

def train_ml_model():
    """
    Train and save the machine learning model
    """
    try:
        # Load the cleaned dataset
        data = pd.read_csv("Cleaned_Patient_Data.csv")
        
        # Separate features and target
        features = data.drop(columns=['Disease_Predictions'])
        target = data['Disease_Predictions']
        
        # One-hot encoding for categorical features
        features = pd.get_dummies(features, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {'n_neighbors': range(1, 21)}
        grid_search = GridSearchCV(
            KNeighborsClassifier(), 
            param_grid, 
            cv=5, 
            scoring='accuracy'
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_k = grid_search.best_params_['n_neighbors']
        best_model = KNeighborsClassifier(n_neighbors=best_k)
        best_model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        with open('disease_prediction_model.pkl', 'wb') as model_file:
            pickle.dump(best_model, model_file)
        
        with open('feature_scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        
        return best_model, scaler
    
    except Exception as e:
        print(f"Error in model training: {e}")
        return None, None

def load_ml_model():
    """
    Load existing ML model or train a new one
    """
    try:
        with open('disease_prediction_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('feature_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        return model, scaler
    
    except FileNotFoundError:
        return train_ml_model()

def home(request):
    """
    Home page view
    """
    return render(request, 'home.html')

def register(request):
    """
    User registration view
    """
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            phone_number = form.cleaned_data.get('phone_number')
            dob = form.cleaned_data.get('dob')
            hospital_name = form.cleaned_data.get('hospital_name')
            
            # Save user profile
            profile = UserProfile(
                user=user,
                phone_number=phone_number,
                dob=dob,
                hospital_name=hospital_name
            )
            profile.save()
            
            login(request, user)
            return redirect('successfully_registered')
    else:
        form = RegisterForm()
    
    return render(request, "register.html", {"form": form})

def login_view(request):
    """
    User login view
    """
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                return redirect('successfully_logged_in')
    else:
        form = AuthenticationForm()
    
    return render(request, 'login.html', {'form': form})

def prediction_form(request):
    """
    Prediction form view
    """
    return render(request, 'prediction_form.html')

def result(request):
    """
    Prediction result view
    """
    try:
        # Load the ML model
        model, scaler = load_ml_model()
        
        # Collect input features from the request
        input_features = [
            float(request.GET.get('n1', 0)),  # Patient ID
            float(request.GET.get('n2', 0)),  # Weight
            float(request.GET.get('n3', 0)),  # Temperature
            float(request.GET.get('n4', 0)),  # Heart rate
            float(request.GET.get('n5', 0)),  # Symptoms
            float(request.GET.get('n6', 0)),  # Existing Condition
            float(request.GET.get('n7', 0)),  # Lab test result
            float(request.GET.get('n8', 0)),  # Cholesterol
            float(request.GET.get('n9', 0)),  # Blood sugar
            float(request.GET.get('n10', 0)),  # Family History
            float(request.GET.get('n11', 0)),  # Smoking
            float(request.GET.get('n12', 0)),  # Systolic BP
            float(request.GET.get('n13', 0)),  # Diastolic BP
            float(request.GET.get('n14', 0))   # BMI
        ]
        
        # Scale the input features
        scaled_features = scaler.transform([input_features])
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Interpret the result
        result2 = "Disease Detected" if prediction[0] == 1 else "No Disease Detected"
        
        # Save patient record (optional)
        Patient.objects.create(
            patient_id=input_features[0],
            prediction_result=result2,
            # Add other relevant fields
        )
        
        # Render the prediction form with the result
        return render(request, 'prediction_form.html', {'result2': result2})
    
    except Exception as e:
        # Handle any errors during prediction
        error_message = f"An error occurred: {str(e)}"
        return render(request, 'prediction_form.html', {'result2': error_message})

def successfully_registered(request):
    """
    Successful registration page
    """
    return render(request, 'successfully_registered.html')

def successfully_logged_in(request):
    """
    Successful login page
    """
    return render(request, 'successfully_logged_in.html')

def history(request):
    """
    View patient prediction history
    """
    patients = Patient.objects.all()
    return render(request, 'history.html', {'patients': patients})