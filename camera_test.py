import serial
import anthropic
import base64
import subprocess

ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
client = anthropic.Anthropic(api_key="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")

def take_photo():
    subprocess.run([
        "rpicam-still", "-o", "item.jpg",
        "--nopreview", "-t", "1000"
    ])

def classify_plastic():
    with open("item.jpg", "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_data
                    }
                },
                {
                    "type": "text",
                    "text": """Look at this plastic item. Identify the plastic type.
                    Reply with ONLY one word: PET, HDPE, PP, or OTHER.
                    No explanation, just the one word."""
                }
            ]
        }]
    )
    return response.content[0].text.strip()

def sort():
    print("📸 Taking photo...")
    take_photo()
    print("🤖 Classifying plastic...")
    plastic_type = classify_plastic()
    print(f"✅ Detected: {plastic_type}")
    ser.write(f"{plastic_type}\n".encode())
    print(f"📤 Sent to Arduino: {plastic_type}")

# Main loop
while True:
    user_input = input("\nPress 1 to scan, q to quit: ")
    if user_input == "1":
        sort()
    elif user_input == "q":
        print("Goodbye!")
        ser.close()
        break
    else:
        print("Invalid input. Press 1 to scan or q to quit.")
