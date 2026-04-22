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
        # Capture image
        frame = camera.capture_array()

        # Save it so we can see what the camera captured
        img = Image.fromarray(frame).resize((224, 224))
        img.save("/home/abdalrahman/Desktop/Raspberry/last_capture.jpg")
        print("📸 Image saved as last_capture.jpg — check what it looks like!")

        # Convert to UINT8
        input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

        # Run the model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get result
        predicted_index = np.argmax(output_data)
        confidence = output_data[predicted_index] / 255.0 * 100
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
