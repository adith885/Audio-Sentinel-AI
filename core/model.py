import tensorflow_hub as hub

# Load YAMNet model
def load_model():
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model