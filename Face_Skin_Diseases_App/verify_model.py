from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("face_skin_disease_mobilenetv2.h5")
classes = ['Acne', 'Actinic keratosis', 'Basal cell carcinoma', 'Eczema', 'Rosacea']

img_path = "Face_Skin_Diseases/test/Acne/032897HB.jpg"  # change this path
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
print("Predicted Disease:", classes[np.argmax(pred)])
