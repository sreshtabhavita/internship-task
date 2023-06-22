import tkinter as tk
from tkinter import filedialog
from tkinter import *
import os
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk

def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def detect_features(file_path):
    global label1, label2, label3, model_eye, model_mouth, model_wrinkle

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    lefteye_2splits = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    wrinkle_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_list = ["No Face", "Face"]
    eye_list = ["Close", "Open"]
    mouth_list = ["Close", "Open"]
    wrinkle_list = ["No Wrinkles", "Wrinkles"]

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) == 0:
        label1.configure(foreground='#011638', text="No Face")
        label2.configure(foreground='#011638', text="No Face")
        label3.configure(foreground='#011638', text="No Face")
        return

    for (x, y, w, h) in faces:
        face_roi = gray_image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi, 1.3, 5)
        eyes_open = False
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_roi, (48, 48))
            eye_img = eye_img[np.newaxis, :, :, np.newaxis]
            eye_pred = eye_list[np.argmax(model_eye.predict(eye_img))]
            if eye_pred == "Open":
                eyes_open = True

        mouth = mouth_cascade.detectMultiScale(face_roi, 1.3, 5)
        mouth_open = False
        for (mx, my, mw, mh) in mouth:
            mouth_roi = face_roi[my:my+mh, mx:mx+mw]
            mouth_img = cv2.resize(mouth_roi, (48, 48))
            mouth_img = mouth_img[np.newaxis, :, :, np.newaxis]
            mouth_pred = mouth_list[np.argmax(model_mouth.predict(mouth_img))]
            if mouth_pred == "Open":
                mouth_open = True

        wrinkles = wrinkle_cascade.detectMultiScale(face_roi, 1.3, 5)
        wrinkles_present = False
        for (wx, wy, ww, wh) in wrinkles:
            wrinkle_roi = face_roi[wy:wy+wh, wx:wx+ww]
            wrinkle_img = cv2.resize(wrinkle_roi, (48, 48))
            wrinkle_img = wrinkle_img[np.newaxis, :, :, np.newaxis]
            wrinkle_pred = wrinkle_list[np.argmax(model_wrinkle.predict(wrinkle_img))]
            if wrinkle_pred == "Wrinkles":
                wrinkles_present = True

        label1.configure(foreground='#011638', text="Eyes: " + ("Open" if eyes_open else "Closed"))
        label2.configure(foreground='#011638', text="Mouth: " + ("Open" if mouth_open else "Closed"))
        label3.configure(foreground='#011638', text="Wrinkles: " + ("Present" if wrinkles_present else "Not Present"))

def select_image():
    global panel, file_path
    file_path = filedialog.askopenfilename()

    if len(file_path) > 0:
        image = Image.open(file_path)
        image = image.resize((250, 250), Image.LANCZOS)
        image = ImageTk.PhotoImage(image)

        if panel is None:
            panel = Label(image=image)
            panel.image = image
            panel.pack(side="left", padx=10, pady=10)
        else:
            panel.configure(image=image)
            panel.image = image

def detect():
    if len(file_path) > 0:
        detect_features(file_path)
    else:
        label1.configure(foreground='#011638', text="No Image Selected")
        label2.configure(foreground='#011638', text="No Image Selected")
        label3.configure(foreground='#011638', text="No Image Selected")

root = tk.Tk()
panel = None
file_path = ''

root.title("FACIAL FEATURE DETECTION")
root.geometry("800x400")

button_select = Button(root, text="Select an image", command=select_image)
button_select.pack(side="top", padx=10, pady=10)

button_detect = Button(root, text="Detect", command=detect)
button_detect.pack(side="top", padx=10, pady=10)

label1 = Label(root, text="")
label1.pack(side="top")

label2 = Label(root, text="")
label2.pack(side="top")

label3 = Label(root, text="")
label3.pack(side="top")

# Define the file paths for the models
model_eye_file = "model_a1.json"
model_eye_weights = "model_weights1.h5"
model_mouth_file = "model_a1.json"
model_mouth_weights = "model_weights1.h5"
model_wrinkle_file = "model_a1.json"
model_wrinkle_weights = "model_weights1.h5"

# Load the models
model_eye = load_model(model_eye_file, model_eye_weights)
model_mouth = load_model(model_mouth_file, model_mouth_weights)
model_wrinkle = load_model(model_wrinkle_file, model_wrinkle_weights)

root.mainloop()
