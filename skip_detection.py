import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
THRESHOLD = 0.5
MIN_DISTANCE = 1000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

plt.ion()
fig, ax = plt.subplots(figsize=(14, 6))
line, = ax.plot(np.zeros(CHUNK), label="Live Audio")
ax.set_ylim(-1, 1)
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")
ax.set_title("Live Skip Detection")
ax.legend()
ax.grid(True)

audio_buffer = np.array([], dtype=np.float32)
skip_count = 0

while True:
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
    audio_buffer = np.concatenate((audio_buffer[-MIN_DISTANCE:], data))
    
    envelope = np.abs(data)
    diff = np.diff(envelope)
    peaks, _ = find_peaks(np.abs(diff), height=THRESHOLD, distance=MIN_DISTANCE)
    
    if len(peaks) > 0:
        skip_count += len(peaks)
        print(f"Detected {len(peaks)} skips. Total skips: {skip_count}")
    
    line.set_ydata(data)
    ax.set_xlim(0, CHUNK)
    fig.canvas.draw()
    fig.canvas.flush_events()

stream.stop_stream()
stream.close()
p.terminate()
plt.ioff()
plt.close()
