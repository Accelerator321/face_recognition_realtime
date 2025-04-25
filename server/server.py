import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Tuple
from sklearn.metrics.pairwise import cosine_similarity


# ENV-based mode config
mode = os.environ.get("mode", "fnet").lower()
prod = os.environ.get("prod", "false").lower() == "true"
debug = not prod


# === Embedding model wrappers ===
class FNET:
    def __init__(self):
        from keras_facenet import FaceNet
        self.model = FaceNet()

    def predict(self, img):
        return self.model.embeddings(img)


# === Face Recognition Module ===
class FRM:
    def __init__(self, model, dim):
        self.model = model
        self.people = {}  # in-memory embedding store
        self.dim = dim

    def add_person(self, name: str, train: np.ndarray) -> None:
        embeddings = []
        if name in self.people:
            embeddings.append(self.people[name])

        img = cv2.resize(train, self.dim)
        img = np.expand_dims(img, axis=0)
        embeddings.append(self.model.predict(img))
        embeddings = np.array(embeddings)
        self.people[name] = np.mean(embeddings, axis=0)

    def del_person(self, name: str):
        if name in self.people:
            del self.people[name]
            return f"Deleted {name} from database."
        return None

    def recognize(self, img: np.ndarray) -> Tuple[float, str]:
        res = [0, None]
        img = cv2.resize(img, self.dim)
        img = np.expand_dims(img, axis=0)
        img = self.model.predict(img)

        for name, features in self.people.items():
            similarity = cosine_similarity(img, features.reshape(1, -1))
            if similarity > res[0]:
                res[0] = similarity
                res[1] = name
        return float(res[0]), res[1]

    def is_allowed(self, img: np.ndarray) -> bool:
        score, _ = self.recognize(img)
        return score * 100 >= 85


# === Init model based on mode ===
if mode == "fnet":
    embedder = FNET()
    face_system = FRM(embedder, dim=(160, 160))
# else:
#     model = tf.keras.models.load_model("my_model.keras")
#     face_system = FRM(model, dim=(64, 64))

# === Flask setup ===
app = Flask(__name__)
CORS(app)

@app.route("/get", methods=["GET"])
def get_status():
    return jsonify({
        "status": "success",
        "message": "Face recognition server is running!",
        "mode": mode,
        "endpoints": {
            "/": "POST - Action router (upload, recognize, delete)",
            "/get": "GET - Server status"
        }
    }), 200

@app.route("/", methods=["POST"])
def home():
    files = request.files.getlist('images')
    action_type = request.form.get('type')

    if action_type == "upload":
        name = request.form.get("name")
        return upload_faces(files, name)

    if action_type == "recognize":
        return recognize_faces(files)

    if action_type == "delete":
        name = request.form.get("name")
        return delete_user(name)

    return jsonify({"status": "error", "message": "Please provide valid 'type' param."}), 400

def upload_faces(files, name):
    try:
        if not name:
            return {'status': 'error', 'message': 'Name is required.'}, 400

        saved = 0
        for file in files:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                face_system.add_person(name, img)
                saved += 1

        return {'status': 'success', 'message': f'{saved} face(s) saved for {name}'}, 200

    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

def recognize_faces(files):
    try:
        results = []
        for idx, file in enumerate(files):
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                results.append({'image': f'image_{idx}', 'name': 'NONE', 'similarity_score': 0.0})
                continue

            score, name = face_system.recognize(img)
            results.append({
                'image': f'image_{idx}',
                'name': name,
                'similarity_score': score
            })

        return {'status': 'success', 'recognized_faces': results}, 200

    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

def delete_user(name):
    msg = face_system.del_person(name)
    if msg:
        return {"status": "success", "message": msg}, 200
    return {"status": "error", "message": "User not found"}, 404


# === For local testing ===
if __name__ == "__main__":
    print(f"🚀 Running in mode: {mode}, debug: {debug}")
    app.run(host="0.0.0.0", port=8000, debug=debug)
