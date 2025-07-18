from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


model = load_model(r"D:\New folder\Desktop\Project8\brain_tumorCNN\brain_tumor_model.h5")

class_labels = ['Glioma_Tumor', 'Meningioma_Tumor', 'No_Tumor', 'Pituitary_Tumor']

img_path = r"D:\New folder\Desktop\Project8\brain_tumorCNN\Testing\pituitary\Te-pi_0010.jpg"   
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize 
img_array = np.expand_dims(img_array, axis=0) 

prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)
predicted_class_name = class_labels[predicted_class_index]


print("Predicted class:", predicted_class_name)
