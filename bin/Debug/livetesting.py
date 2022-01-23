import pickle  # to save model after training
import wave
from array import array
from struct import pack
from sys import byteorder

import librosa
import numpy as np
import pyaudio
import soundfile  # to read audio file
import speech_recognition as sr
import matplotlib.pyplot as plt
from scipy.io import wavfile


def extract_feature(file_name, **kwargs):

    """
        Extract feature from audio file `file_name`
    Features supported:
                - MFCC (mfcc)
                - Chroma (chroma)
                - MEL Spectrogram Frequency (mel)
                - Contrast (contrast)
                - Tonnetz (tonnetz)
            e.g:
            `features = extract_feature(path, mel=True, mfcc=True)`
        """

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        x = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(x))
        result_feat = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
            result_feat = np.hstack((result_feat, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result_feat = np.hstack((result_feat, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(x, sr=sample_rate).T, axis=0)
            result_feat = np.hstack((result_feat, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result_feat = np.hstack((result_feat, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate).T, axis=0)
            result_feat = np.hstack((result_feat, tonnetz))
    return result_feat


# from utils import extract_feature

THRESHOLD = 200
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 1
SILENCE = 5  # поменял с 30

RECORD_SECONDS = 5


def is_silent(snd_data):
    """Returns 'True' if below the 'silent' threshold"""
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    """Average the volume out"""
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    """Trim the blank spots at the start and end"""
    def _trim(snd_data_trim):
        snd_started = False
        r = array('h')
        for i in snd_data_trim:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    """
    Add silence to the start and end of 'snd_data'
    of length 'seconds' (float)
    """
    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r


def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    time = 0
    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

        time += 1
        if time == 50:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    """
    Records from the microphone and outputs
    the resulting data to 'path'
    """
    sample_width, record_data = record()
    record_data = pack('<' + ('h' * len(record_data)), *record_data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(record_data)
    wf.close()


def main():
    # if __name__ == "__main__":
    # def int():
        # load the saved model (after training)
    loaded_model = pickle.load(open("mlp_classifier.model", 'rb'))
    please1 = 'Please talk'
    print(please1)
    filename = "test.wav" # запись
    a = filename
    # record the file (start talking)
    record_to_file(filename)
    # -----------Speech to text------------
    mic = sr.Recognizer()
    # open the file
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = mic.record(source)
        # extract features and reshape it
        features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        # predict
        result = loaded_model.predict(features)[0]
        # show the result !
        emotion = "Predicted emotion is: " + result
        print(emotion)
        # recognize (convert from speech to text)
        text = mic.recognize_google(audio_data)
        # print(text)

    rate, data = wavfile.read(filename)

    return emotion, text, data


# emotion, data = main()

# plt.plot(data)
# plt.show()
# plt.savefig('wave.png')  # to save the ploting figure
