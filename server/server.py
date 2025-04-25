import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras_facenet import FaceNet
from pymongo import MongoClient
# ENV-based mode config
mode = os.environ.get("mode", "fnet").lower()
prod = os.environ.get("prod", "false").lower() == "true"
debug = not prod


# === Embedding model wrappers ===


embedder = FaceNet()
class FNET:
  def __init__(self, model):
    self.model = model
  def predict(self,img):
    embedding = self.model.embeddings(img)
    return embedding



class FRM:
    def __init__(self, model, dim,
                 mongo_uri=os.environ.get("db_url"),
                 db_name="face_db"):
        self.model = model
        self.dim = dim

        # Setup Mongo connection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["embeddings"]

        self.people = self.read_db()

    def read_db(self) -> dict:
        """Reads embeddings from MongoDB and returns them as a dictionary."""
        people = {}
        for doc in self.collection.find():
            people[doc["name"]] = np.array(doc["embedding"])
        return people

    def save_db(self) -> None:
        """Saves current in-memory embeddings to MongoDB."""
        self.collection.delete_many({})
        docs = [{"name": k, "embedding": v.tolist()} for k, v in self.people.items()]
        if docs:
            self.collection.insert_many(docs)

    def add_person(self, name: str, train: List[np.ndarray]) -> None:
        """
        Adds a new person or updates existing person with embeddings
        averaged over multiple training images.
        """
        # Resize and stack images properly into shape (n, h, w, c)
        processed_imgs = [cv2.resize(img, self.dim) for img in train]
        imgs_array = np.stack(processed_imgs, axis=0)

        # # Get embeddings
        embeddings = self.model.predict(imgs_array)
        mean_embedding = np.mean(embeddings, axis=0)

        # # If the person already exists, average with old embedding
        if name in self.people:
            mean_embedding = np.mean([self.people[name], mean_embedding], axis=0)

        self.people[name] = mean_embedding
        self.save_db()

    def del_person(self, name: str):
        if name in self.people:
            del self.people[name]
            self.save_db()
            return f"Deleted person {name} from DB"
        return None

    def recognize(self, img: np.ndarray) -> Tuple[float, str]:
        res = [0, None]
        img = cv2.resize(img, self.dim)
        img = np.expand_dims(img, axis=0)
        img_embedding = self.model.predict(img)
        for name, features in self.people.items():
            similarity = cosine_similarity(img_embedding, features.reshape(1, -1))[0][0]
            if similarity > res[0]:
                res[0] = similarity
                res[1] = name
        return res

    def is_allowed(self, img: np.ndarray) -> bool:
        res = self.recognize(img)
        return res[0] * 100 >= 85

    def test(self, home_group: dict, test_batch: dict):
        """
        home_group: dict of {name: List[np.ndarray]} — authorized people
        test_batch: dict of {name: List[np.ndarray]} — test images from both authorized and others
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_true = []
        y_pred = []

        authorized_names = set(home_group.keys())

        for name, imgs in test_batch.items():
            label = 1 if name in authorized_names else 0  # 1: should be allowed, 0: should be denied

            for img in imgs:
                prediction = self.is_allowed(img)
                y_true.append(label)
                y_pred.append(int(prediction))  # convert bool to int

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }


    def recognize_batch(self, images: List[np.ndarray]) -> Tuple[dict, dict]:
        score_tracker = []
        name_counter = {}
        max_score = 0
        max_score_name = None

        for img in images:
            score, name = self.recognize(img)
            score_tracker.append((score, name))

            if score > max_score:
                max_score = score
                max_score_name = name

            if name:
                name_counter.setdefault(name, []).append(score)

        avg_name, avg_score = None, 0
        for name, scores in name_counter.items():
            mean_score = np.mean(scores)
            if mean_score > avg_score:
                avg_score = mean_score
                avg_name = name

        return (
            {"max_score": float(max_score), "name": max_score_name},
            {"avg_score": float(avg_score), "name": avg_name}
        )



# === Init model based on mode ===
if mode == "fnet":
    model = FNET(embedder)
    face_system = FRM(model, dim=(160, 160),db_name = "face_db_facenet")
else:
    model = tf.keras.models.load_model("my_model.keras")
    face_system = FRM(model, dim=(64, 64),db_name = "face_db_cnn")

# === Flask setup ===
app = Flask(__name__)
CORS(app)





@app.route("/get", methods=["GET"])
def get():
    return jsonify({
        "status": "success",
        "message": "🚀 Flask face-upload server is running!",
        "endpoints": {
            "/": "GET - Server status and available routes",
            "/upload_faces": "POST - Upload one or more face images (form-data key: 'images')"
        }
    }), 200

@app.route('/', methods=['POST'])
def home():
    files = request.files.getlist('images')
    type = request.form.get('type')
    if type == "upload":
        name = request.form.get('name')
        return upload_faces(files, name)
    if type == "recognize":
        return recognize_faces()
    if type == "delete":
        name = request.form.get('name')
        return delete_user(name)
    return "Please mention a type", 200

def delete_user(name):
  res = face_system.del_person(name)
  if res is None:
    return {"status": "error", "message": "User not found"},404

  return {"status": "success", "message": res},200

def upload_faces(files, name):
    try:
        if not name:
            return {'status': 'error', 'message': 'Name is required.'}, 400

        images = []
        saved_faces = []

        for idx, file in enumerate(files):
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                images.append(img)
                saved_faces.append(f"{name}_{idx}.jpg")

        if images:
            face_system.add_person(name, images)
        else:
            return {'status': 'error', 'message': 'No valid images.'}, 400

        return {'status': 'success', 'saved_faces': saved_faces, "name": name}, 200

    except Exception as e:
        print(e)
        return {'status': 'error', 'message': str(e)}, 500

def recognize_faces():
    try:
        files = request.files.getlist('images')
        if not files:
            return {'status': 'error', 'message': 'No images provided.'}, 400

        images = []
        for file in files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
            else:
                images.append(None)

        results = face_system.recognize_batch(images)
        

        return {'status': 'success', 'recognized_faces': results}, 200

    except Exception as e:
        print(e)
        return {'status': 'error', 'message': str(e)}, 500

# === For local testing ===
if __name__ == "__main__":
    print(f"🚀 Running in mode: {mode}, debug: {debug}")
    port = int(os.environ.get("PORT", 10000))  # fallback is optional
    app.run(host="0.0.0.0", port=port, debug=debug)
