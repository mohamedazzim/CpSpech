import pyttsx3

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[ERROR] TTS engine failed: {e}")

if __name__ == "__main__":
    import sys
    speak(" ".join(sys.argv[1:]))
