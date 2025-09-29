import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template

# --- 1. SETUP ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join("model", "dgcnn_malware_detector.h5")
NUM_API_CALLS = 307
SEQUENCE_LENGTH = 100

# --- 2. CUSTOM KERAS LAYER ---
class GraphConvLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GraphConvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[1][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(feature_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(GraphConvLayer, self).build(input_shape)

    def call(self, inputs):
        adj_matrix, feat_matrix = inputs
        support = tf.matmul(adj_matrix, feat_matrix)
        output = tf.matmul(support, self.kernel)
        return output

    def get_config(self):
        config = super(GraphConvLayer, self).get_config()
        config.update({"output_dim": self.output_dim})
        return config

# --- 3. LOAD MODEL ---
try:
    print("Loading DGCNN model...")
    model = load_model(MODEL_PATH, custom_objects={'GraphConvLayer': GraphConvLayer})
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- 4. PREPROCESSING FUNCTIONS ---
def sequence_to_graph(api_sequence):
    adj_matrix = np.zeros((NUM_API_CALLS, NUM_API_CALLS), dtype=np.float32)
    for i in range(len(api_sequence) - 1):
        u, v = int(api_sequence[i]), int(api_sequence[i+1])
        adj_matrix[u, v] = 1.0
    return adj_matrix

def extract_api_sequence_from_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read().strip()
        api_sequence = np.array(
            [int(x) for x in content.split() if x.isdigit() and 0 <= int(x) < NUM_API_CALLS],
            dtype=int
        )
        if len(api_sequence) > SEQUENCE_LENGTH:
            api_sequence = api_sequence[:SEQUENCE_LENGTH]
        elif len(api_sequence) < SEQUENCE_LENGTH:
            padding = np.zeros(SEQUENCE_LENGTH - len(api_sequence), dtype=int)
            api_sequence = np.concatenate([api_sequence, padding])
    return api_sequence

# --- 5. ROUTES ---

# Render the front-end template
@app.route("/")
def index():
    return render_template("index.html")

# File analysis endpoint
@app.route("/analyze", methods=["POST"])
def analyze_file():
    if model is None: 
        return jsonify({"error": "Model not loaded properly."}), 500
    if 'file' not in request.files: 
        return jsonify({"error": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '': 
        return jsonify({"error": "No file selected"}), 400

    temp_filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(temp_filepath)

        api_sequence = extract_api_sequence_from_file(temp_filepath)
        graph = sequence_to_graph(api_sequence)

        # Add batch dimension
        graph_input = np.array([graph])
        sequence_input = np.array([api_sequence])

        prediction_array = model.predict([graph_input, sequence_input])
        prediction_prob = float(prediction_array[0][0])
        prediction_label = "Malware" if prediction_prob > 0.5 else "Goodware"

        return jsonify({
            "prediction": prediction_label,
            "confidence_score": f"{prediction_prob:.4f}"
        }), 200

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# --- 6. RUN SERVER ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
