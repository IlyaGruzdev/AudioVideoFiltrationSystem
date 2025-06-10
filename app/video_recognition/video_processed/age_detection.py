import cv2
import numpy as np
import tensorflow as tf
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
def mae(y_true, y_pred):
  return tf.keras.metrics.mean_absolute_error(y_true, y_pred)
class AgePredictor:
    def __init__(self, model_path = cur_path+"/models/age_gender_model.h5"):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'mae': mae})
        self.input_shape = self.model.input_shape[1:4]
 
    def predict_age(self, face_image):
        face = cv2.resize(face_image, (128, 128))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0

        if self.input_shape[-1] == 1:
            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        face = np.expand_dims(face, axis=0)

        predictions = self.model.predict(face, verbose=0)
        if isinstance(predictions, list) and len(predictions) >= 2:
            age = float(predictions[1][0][0])
        else:
            age = float(predictions[0][0])
        return int(round(age))

