from picamera2 import Picamera2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

# Load model
interpreter = tflite.Interpreter(model_path="/home/pi/plastic_sorter/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("/home/pi/plastic_sorter/labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

# Setup camera (same as your working code)
camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.start()
time.sleep(2)

print("Plastic Sorter Ready! Press 1 + Enter to scan, or q to quit.")

while True:
    user_input = input("> ").strip()

    if user_input == "1":
        # Capture image (same as your working code)
        frame = camera.capture_array()

        # Resize to 224x224 (what the model expects)
        img = Image.fromarray(frame).resize((224, 224))
        input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        # Run the model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get result
        predicted_index = np.argmax(output_data)
        confidence = output_data[predicted_index] * 100
        label = labels[predicted_index]

        print(f"🔍 Result: {label.upper()} ({confidence:.1f}% confidence)")

        if confidence < 60:
            print("⚠️  Low confidence — try better lighting or reposition the plastic.\n")

    elif user_input.lower() == "q":
        print("Exiting...")
        camera.stop()
        break

    else:
        print("Invalid input. Press 1 to scan or q to quit.")