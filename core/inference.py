import numpy as np

# Load class labels correctly
def load_class_names(model):
    path = model.class_map_path().numpy().decode('utf-8')

    class_names = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            class_names.append(parts[2])  # correct label

    return class_names


def predict(model, waveform, class_names):
    scores, embeddings, spectrogram = model(waveform)

    scores = np.mean(scores.numpy(), axis=0)

    top_index = np.argmax(scores)

    return class_names[top_index], float(scores[top_index]), scores