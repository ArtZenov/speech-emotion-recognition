import librosa
import soundfile  # считать аудиофайл
import numpy as np
import glob
import os
import pickle  # сохранение учебной модели
from sklearn.model_selection import train_test_split  # разделение учебной и тестовой модели
from sklearn.neural_network import MLPClassifier  # многослойная модель персептрона
from sklearn.metrics import accuracy_score   # для измерения точности вычислений
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def extract_feature(file_name, **kwargs):
    """
        Extract feature from audio file `file_name`
            Features supported:
                - MFCC (mfcc) - Кепстральные коэффициенты Mel-частоты
                - Chroma (chroma) -
                - MEL Spectrogram Frequency (mel)
                - Contrast (contrast)
                - Tonnetz (tonnetz)
            e.g.:
            `features = extract_feature(path, mel=True, mfcc=True)`
    """

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


def load_data(test_size=0.2):
    # все доступные эмоции датасета RAVDESS
    int2emotion = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }

    # эмоции, которые были использованы
    AVAILABLE_EMOTIONS = {
        "angry",
        "sad",
        "neutral",
        "happy"}

    X, y = [], []
    for file in glob.glob("C:/Users/Test/Google Диск/МГЛУ работа/Научная лаборатория/Проект SER/SER 3.6.6/data/Actor_*/*.wav"):
        # получение имени аудиофайла
        basename = os.path.basename(file)
        # получение метки эмоции
        emotion = int2emotion[basename.split("-")[2]]
        # гейт для разрешения только использованных эмоций
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # извлекаем особенности речи
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # добавляем данные
        X.append(features)
        y.append(emotion)
    # разделяем данные на учебные и тестовые и возвращаем
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
# загружаем датасет RAVDESS, 75% учебная 25% тестовая


X_train, X_test, y_train, y_test = load_data(test_size=0.25)

# print("Features extracted: ", len(X_train))
print("Number of training samples: ", len(y_train))
print("Number of testing samples: ", len(y_test))
# print("Number of features: ", len(X_test))


# лучшая модель, определенная поиском по сетке
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 600,
}


# инициализируем классификатор Multi-Layer Perceptron
# с лучшими параметрами
model = MLPClassifier(**model_params)
print(model)

# учим модель
print("[*] Training the model...")
model.fit(X_train, y_train)

# спрогнозируем 25% данных, чтобы измерить, насколько хороша выборка
y_pred = model.predict(X_test)

# расчёт точности
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
print(classification_report(y_test, y_pred))
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

pickle.dump(model, open("mlp_classifier.model", "wb"))
