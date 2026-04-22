from picamera2 import Picamera2
import time

camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.start()
time.sleep(2)  # warm-up time

print("Camera ready! Press 1 + Enter to take a photo, or q + Enter to quit.")

photo_count = 1

while True:
    user_input = input("> ").strip()

    if user_input == "1":
        filename = f"photo_{photo_count}.jpg"
        camera.capture_file(filename)
        print(f"📸 Photo saved as {filename}")
        photo_count += 1

    elif user_input.lower() == "q":
        print("Exiting...")
        camera.stop()
        break

    else:
        print("Invalid input. Press 1 to take a photo, or q to quit.")