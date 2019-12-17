import wave

import numpy as np

from linalg_project.preprocessing import DEFAULT_RATE, audio, SOUND_FORMAT, CHANNELS, DEFAULT_CHUNK, NUMPY_FORMAT, \
    SAMPLE_WIDTH


def record_mic(seconds=5, sample_rate=DEFAULT_RATE):
    frames = []
    stream = audio.open(format=SOUND_FORMAT, channels=CHANNELS, rate=sample_rate, input=True, frames_per_buffer=CHUNK)
    for i in range(0, int(sample_rate / DEFAULT_CHUNK * seconds)):
        data = stream.read(DEFAULT_CHUNK)
        frames.append(data)
    data_str = (b"").join(frames)
    return np.frombuffer(data_str, dtype=NUMPY_FORMAT)


def gen_sound(np_arr, play=False, saveto=None, sample_rate=DEFAULT_RATE):
    data = np_arr.astype("int16").tobytes()
    if saveto is not None:
        f = wave.open(saveto, "wb")
        f.setnchannels(CHANNELS)
        f.setsampwidth(SAMPLE_WIDTH)
        f.setframerate(sample_rate)
        f.writeframes(data)
        f.close()
    if play:
        stream = audio.open(format=SOUND_FORMAT, channels=CHANNELS, rate=sample_rate, output=True)
        for i in range(0, len(data), DEFAULT_CHUNK):
            chunk = data[i: i + DEFAULT_CHUNK]
            stream.write(chunk)