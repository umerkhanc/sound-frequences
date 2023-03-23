import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# Set the parameters for the audio stream
chunk_size = 1024  # number of audio samples per chunk
sample_rate = 44100  # sample rate in Hz

# Initialize the audio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

# Initialize the plot
fig, ax = plt.subplots()
freqs = np.fft.fftfreq(chunk_size, 1.0 / sample_rate)
line, = ax.plot(freqs, np.zeros(chunk_size))

# Continuously update the plot with the live sound frequencies
while True:
    # Read a chunk of audio data from the stream
    data = stream.read(chunk_size)

    # Convert the audio data to a numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Compute the one-dimensional discrete Fourier Transform
    frequencies = np.fft.fft(audio_data)

    # Update the line in the plot with the new frequencies
    line.set_ydata(np.abs(frequencies))
    ax.set_ylim(0, np.max(np.abs(frequencies)))

    # Redraw the plot
    fig.canvas.draw()
    plt.pause(0.001)