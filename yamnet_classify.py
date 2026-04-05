"""YAMNet TFLite audio classifier for dog bark detection.

Uses the YAMNet model (521 AudioSet classes) to classify short audio clips.
Dog-related classes (indices 70-75): Bark, Yip, Howl, Bow-wow, Growling, Whimper.
"""

import csv
import os

import numpy as np
import tflite_runtime.interpreter as tflite
from scipy.io import wavfile
from scipy.signal import resample

_interpreter = None
_class_names = None

MODEL_PATH = os.environ.get("YAMNET_MODEL_PATH", "/app/yamnet.tflite")
CLASS_MAP_PATH = os.environ.get("YAMNET_CLASS_MAP_PATH", "/app/yamnet_class_map.csv")

YAMNET_SAMPLE_RATE = 16000
DOG_CLASS_INDICES = {70, 71, 72, 73, 74, 75}
DOG_CONFIDENCE_THRESHOLD = 0.25


def _load_model():
    global _interpreter
    if _interpreter is not None:
        return _interpreter
    _interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    _interpreter.allocate_tensors()
    return _interpreter


def _load_class_names():
    global _class_names
    if _class_names is not None:
        return _class_names
    _class_names = {}
    with open(CLASS_MAP_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            idx = int(row[0])
            display_name = row[2]
            _class_names[idx] = display_name
    return _class_names


def _read_wav_as_float(wav_path):
    """Read a WAV file and return 16kHz mono float32 samples in [-1, 1]."""
    sample_rate, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if sample_rate != YAMNET_SAMPLE_RATE:
        num_samples = int(len(data) * YAMNET_SAMPLE_RATE / sample_rate)
        data = resample(data, num_samples).astype(np.float32)

    return data


def is_dog_barking(wav_path):
    """Classify audio and check for dog-related sounds.

    Returns (is_dog, top_class_name, top_confidence, dog_confidence).
    """
    interpreter = _load_model()
    class_names = _load_class_names()
    waveform = _read_wav_as_float(wav_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    expected_samples = input_shape[0]

    if len(waveform) < expected_samples:
        waveform = np.pad(waveform, (0, expected_samples - len(waveform)))

    num_windows = len(waveform) // expected_samples
    if num_windows == 0:
        num_windows = 1

    all_scores = []
    for i in range(num_windows):
        start = i * expected_samples
        chunk = waveform[start : start + expected_samples]
        if len(chunk) < expected_samples:
            chunk = np.pad(chunk, (0, expected_samples - len(chunk)))
        chunk = chunk.astype(np.float32)

        interpreter.resize_tensor_input(input_details[0]["index"], chunk.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]["index"], chunk)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[0]["index"])
        all_scores.append(scores)

    avg_scores = np.mean(np.concatenate(all_scores, axis=0), axis=0)

    top_idx = int(np.argmax(avg_scores))
    top_class = class_names.get(top_idx, f"class_{top_idx}")
    top_confidence = float(avg_scores[top_idx])

    dog_confidence = float(max(avg_scores[i] for i in DOG_CLASS_INDICES if i < len(avg_scores)))

    is_dog = dog_confidence >= DOG_CONFIDENCE_THRESHOLD

    return is_dog, top_class, top_confidence, dog_confidence
