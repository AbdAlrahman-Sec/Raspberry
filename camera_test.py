from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter
import numpy as np
from PIL import Image
import time

# Load model
interpreter = Interpreter(model_path="/home/abdalrahman/Desktop/Raspberry/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("/home/abdalrahman/Desktop/Raspberry/labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

# Setup camera
camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.start()
time.sleep(2)

print("Plastic Sorter Ready! Press 1 + Enter to scan, or q to quit.")

while True:
    user_input = input("> ").strip()

    if user_input == "1":
        frame = camera.capture_array()
        img = Image.fromarray(frame).resize((224, 224))
        img.save("/home/abdalrahman/Desktop/Raspberry/last_capture.jpg")
        print("📸 Image saved!")

        input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Print ALL scores so we can see what's happening
        print("--- All class scores ---")
        for i, label in enumerate(labels):
            confidence = output_data[i] / 255.0 * 100
            print(f"  {label}: {confidence:.1f}%")
        print("------------------------")

        predicted_index = np.argmax(output_data)
        confidence = output_data[predicted_index] / 255.0 * 100
        label = labels[predicted_index]
        print(f"🔍 Result: {label.upper()} ({confidence:.1f}% confidence)\n")

    elif user_input.lower() == "q":
        print("Exiting...")
        camera.stop()
        break

    else:
        print("Invalid input. Press 1 to scan or q to quit.")
