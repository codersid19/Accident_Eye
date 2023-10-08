import torch
import torchvision
from torchvision import transforms
import cv2
import random
import geocoder
import numpy as np
from PIL import Image
from data_preparation import train_dataset
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
import pytz
import requests
from geopy.geocoders import Nominatim
import geopy.geocoders
from geopy.geocoders import Nominatim
from geopy.geocoders import options


# Your email and password
sender_email = 'siddhant19shirodkar12@gmail.com'
sender_password = 'mbch opea jivw owir'

# Recipient email address
recipient_email = 'shirodkar.siddhant19@gmail.com'

# Fetch current time and date
current_time = datetime.datetime.now(pytz.timezone('America/New_York'))
current_date = current_time.strftime("%Y-%m-%d %H:%M:%S")

loc = ['me', '99.169.28.21', '98.249.182.244']
n = random.randint(0, 2)
location = geocoder.ip(loc[n])
print(location)

# # Fetch current location (latitude and longitude)
# geopy.geocoders.options.default_ssl_context = None
# geolocator = Nominatim(user_agent="My-Application", scheme="http")
# location = geolocator.geocode("Florida")

if location:
    current_location = location
else:
    current_location = None


subject = 'Alert: Accident detected. Emergency Services Needed'
message_body = f"An accident has occurred. Need Emergency services. More details below.\n\n"
message_body += f"Time: {current_time}\n"
message_body += f"Date: {current_date}\n"
if current_location:
    message_body += f"Current Location: {location}\n"
else:
    message_body += "Location not found.\n"


msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = recipient_email
msg['Subject'] = subject
msg.attach(MIMEText(message_body, 'plain'))

model = torchvision.models.resnet152(weights='DEFAULT')
checkpoint = torch.load('S:/Projects/Data Science And ML/Accident Detection/Code/models/myModel.pth')
model.load_state_dict(checkpoint)
model.eval()

class_name = train_dataset.classes
print(class_name)


def predict_frame(frame, model_name, class_names):
    # Apply preprocessing transformations to the frame
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model_name(image)

    _, predicted_class = output.max(1)
    return class_names[predicted_class[0]]


# Open the video file (change 'video_path' to the path of your video file)
video_path = 'S:/Projects/Data Science And ML/Accident Detection/Code/data/part1(split-video.com).mp4'

cap = cv2.VideoCapture(video_path)


accident_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        break  # End of video

    cv2.imshow('Video', frame)

    # Perform prediction on the frame
    predicted_class = predict_frame(frame, model, class_name)


    if predicted_class == 'Accident':
        # Display the frame with the predicted class
        cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Video', frame)
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            server.quit()
            print(f"Email sent to {recipient_email}")
        except Exception as e:
            print(f"Failed to send email: {str(e)}")

    if predicted_class == 'Accident':
        accident_detected = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print the final result
if accident_detected:
    print("Accident occurred in the video.")
else:
    print("No accident detected in the video.")

