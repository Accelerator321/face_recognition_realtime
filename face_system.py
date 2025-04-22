from typing import List,Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

import os
import cv2
import tensorflow as tf

# Disable GPU usage
# tf.config.set_visible_devices([], 'GPU')

# # Verify if GPU is disabled
# physical_devices = tf.config.list_physical_devices('GPU')
# if not physical_devices:
#     print("GPU is disabled")
# else:
#     print(f"Number of GPUs available: {len(physical_devices)}")
# # Load the model
# model = tf.keras.models.load_model('D:\\face_sys\\models\\my_model_tf.h5')

# # Summarize the model
# print(model.summary())

# class FRM:

#     def __init__(self, model, dim):
#         self.model = model
#         if os.path.exists("embedings.json"):
#           with open("embedings.json", "r") as f:
#               loaded = json.load(f)
#               self.people = {k: np.array(v) for k, v in loaded.items()}
#         else:
#             self.people = {}

#         self.dim = dim

#     def add_person(self, name: str, train: np.ndarray)->None:
#         embeddings = []
#         # if name in self.people: return
#         if name in self.people: embeddings.append(self.people[name])
#         img = cv2.resize(train,self.dim)
#         # print(img.shape)

#         img = np.expand_dims(img, axis=0)

#         embeddings.append(self.model.predict(img))


#         embeddings = np.array(embeddings)

#         embeddings  = np.mean(embeddings,axis=0)
#         self.people[name] = embeddings
#         with open("embedings.json","w") as f:
#           data = {k:v.tolist() for k,v in self.people.items()}

#           json.dump(data,f)

#     def recognize(self, img:np.ndarray)->Tuple[int,str]:
#         res = [0,None]
#         img = cv2.resize(img,self.dim)
#         img = np.expand_dims(img, axis=0)


#         img = self.model.predict(img)

#         for name, features in self.people.items():

#             similarity = cosine_similarity(img, features)
#             if similarity  > res[0]:
#                 res[0] = similarity
#                 res[1] = name
#         return res

#     def is_allowed(self, img:np.ndarray)->bool:
#       res = self.recognize(img)
#       if res[0]*100 >= 85:
#         return True
#       else: return False

#     def test(self,x_test, y_test):
#       y_pred = []
#       for img in x_test:
#           ans = self.is_allowed(img)
#           y_pred.append(ans)
#       y_pred = np.array(y_pred)

#       from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#       res= {}
#       res["accuracy"] = accuracy_score(y_test, y_pred)
#       res["precision"] = precision_score(y_test, y_pred)
#       res["recall"] = recall_score(y_test, y_pred)
#       res["f1"] = f1_score(y_test, y_pred)

#       return res
    
# face_system = FRM(model,(64,64))
