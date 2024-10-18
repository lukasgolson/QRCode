import sys
import keras
import onnx
import tensorflow as tf
import tf2onnx
from tensorflow import TensorSpec

def convert_model(tf_model_path, onnx_model_path):
    # Load the TensorFlow model
    model = keras.models.load_model(tf_model_path)

    # Get the input shape
    input_shape = model.input_shape[1:]  # Exclude the batch size

    # Specify the input signature
    input_signature = [TensorSpec(shape=(None, *input_shape), dtype=tf.float32)]

    # Convert the TensorFlow model to ONNX
    onnx_model = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=11)

    # Save the ONNX model
    onnx.save_model(onnx_model, onnx_model_path)  # Use the updated function

    print(f"Model converted and saved to {onnx_model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_tf_to_onnx.py <path_to_tf_model> <output_path_for_onnx_model>")
        sys.exit(1)

    tf_model_path = sys.argv[1]
    onnx_model_path = sys.argv[2]

    convert_model(tf_model_path, onnx_model_path)
