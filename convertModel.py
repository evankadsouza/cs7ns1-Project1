import tensorflow as tf
from tensorflow import keras

# Load the model architecture from JSON
def load_model_from_json(json_file, weights_file):
    with open(json_file, 'r') as json_file:
        model_json = json_file.read()
    model = keras.models.model_from_json(model_json)
    
    # Load weights into the model
    model.load_weights(weights_file)
    
    return model

# Convert the Keras model to TFLite format
def convert_to_tflite(model, tflite_model_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model to a .tflite file
    with open(tflite_model_file, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved as: {tflite_model_file}")

def main():
    # Specify the file names
    json_file = 'test.keras.json'      # Your model architecture JSON file
    weights_file = 'test.keras.keras'  # Your model weights file
    tflite_model_file = 'model.tflite'  # Output TFLite model file

    # Load the model
    model = load_model_from_json(json_file, weights_file)

    # Convert to TFLite
    convert_to_tflite(model, tflite_model_file)

if __name__ == "__main__":
    main()
