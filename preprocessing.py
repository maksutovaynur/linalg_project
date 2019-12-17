import time

import numpy as np
import pyaudio

DEFAULT_RATE = 44100
DEFAULT_CHUNK = 1024
SAMPLE_WIDTH = 2
SOUND_FORMAT = pyaudio.paInt16
NUMPY_FORMAT = np.int16
CHANNELS = 1

audio = pyaudio.PyAudio()


def sleep(seconds):
    time.sleep(seconds)


def gen_time(seconds=5, sample_rate=DEFAULT_RATE):
    return np.arange(0, seconds, 1. / sample_rate)


def gen_sin(freq=440, seconds=5):
    x = gen_time(seconds=seconds)
    return np.sin(2 * np.pi * x * freq)


def gen_saw(freq=440, seconds=5, symmetry=1., sample_rate=DEFAULT_RATE):
    period = 1. / freq
    frame_samples = sample_rate * period
    frame_samples_int = int(frame_samples)
    separator_int = int(frame_samples * symmetry)
    x = gen_time(seconds)
    value = 0. * x[:frame_samples_int]
    value[:separator_int] = 2 * x[:separator_int] / x[separator_int - 1] - 1
    value[separator_int:] = 2 * (period - x[separator_int:frame_samples_int]) / (period - x[separator_int]) - 1
    value = np.expand_dims(value, axis=0)
    return np.squeeze(np.hstack((int(seconds * freq) + 1) * [value]))[:len(x)]


def create_mixture_matrix(a1, b1, a2, b2):
    return np.array([[a1, b1], [a2, b2]])


def mix_signals(matrix, *signals):
    if not isinstance(signals, np.ndarray):
        signals = np.vstack(signals)
    return matrix @ signals


def demix_signals(matrix, *mixed_signals):
    if not isinstance(mixed_signals, np.ndarray):
        mixed_signals = np.vstack(mixed_signals)
    return np.linalg.inv(matrix) @ mixed_signals


def center(signal):
    return signal - np.expand_dims(signal.mean(axis=1), axis=1)


def white(signal):
    matrix = np.cov(signal)
    w, E = np.linalg.eig(matrix)
    D = np.sqrt(np.linalg.pinv(np.diag(w)))
    M = E @ D @ E.T
    return M @ signal


def preprocess(signal):
    return white(center(signal))