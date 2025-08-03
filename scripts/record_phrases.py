import pyaudio
import wave
import os

def record_sample(phrase, sample_idx, duration=3, output_dir="dataset"):
    fs = 16000
    channels = 1
    chunk = 1024

    p = pyaudio.PyAudio()

    dir_path = os.path.join(output_dir, phrase)
    os.makedirs(dir_path, exist_ok=True)

    filename = os.path.join(dir_path, f"sample_{sample_idx:03d}.wav")

    try:
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=fs, input=True, frames_per_buffer=chunk)
        print(f"Recording sample #{sample_idx} for phrase '{phrase}'. Speak now...")
        frames = []
        for _ in range(0, int(fs / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
    except Exception as e:
        print(f"[ERROR] Failed to record audio: {e}")
        return
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()

    try:
        wf = wave.open(filename, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
        wf.close()
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save audio file: {e}")

if __name__ == "__main__":
    phrase = input("Enter phrase label to record: ")
    start_idx = int(input("Start sample index (e.g., 1): "))
    num_samples = int(input("Number of samples to record: "))

    for i in range(start_idx, start_idx + num_samples):
        input(f"Press Enter to start recording sample #{i}...")
        record_sample(phrase, i)
