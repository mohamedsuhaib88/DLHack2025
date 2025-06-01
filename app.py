# ~~~ Importing the libraries ~~~

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Initialize global variables
dgrs = []
res = " "

# ~~~ Prediction Logic ~~~
def predict():
    global res, dgrs
    dgrs.clear()  # Reset previous predictions
    res = " "     # Reset result string

    model = load_model('mnist2.h5')  # Load trained CNN model

    image_folder = "./"
    filename = 'img.jpg'

    # Load the drawn image
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    th = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Find contours
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1).astype("float32") / 255.0
        pred = model.predict(digit)[0]
        final_pred = np.argmax(pred)

        dgrs.append(int(final_pred))
        res += str(final_pred) + " "

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        cv2.putText(image, data, (x, y), font, fontScale, color, thickness)

    # Optionally save or display annotated image
    cv2.imwrite("annotated_result.jpg", image)

# ~~~ UI interface ~~~

# App title
st.title("🖌️ Handwritten Digit Recognition - MNIST")

# Instruction
st.markdown("""
Draw one or more digits on the canvas. Click **Predict** to recognize the digits.
""")

# Canvas
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color='red',
    background_color='black',
    height=150,
    width=400,
    drawing_mode='freedraw',
    key="canvas"
)

# Save canvas content to an image file
if canvas_result.image_data is not None:
    # Convert float32 canvas image to uint8 for OpenCV
    img_uint8 = (canvas_result.image_data).astype(np.uint8)
    cv2.imwrite("img.jpg", img_uint8)

# Prediction button
if st.button("Predict"):
    predict()
    st.success(f"The predicted digit(s): {res.strip()}")
    st.image("annotated_result.jpg", caption="Detected Digits", use_column_width=True)
