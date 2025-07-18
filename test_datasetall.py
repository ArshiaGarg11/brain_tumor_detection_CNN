import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import random


model = load_model(r"D:\New folder\Desktop\Project8\brain_tumorCNN\brain_tumor_model.h5")


test_dir = r"D:\New folder\Desktop\Project8\brain_tumorCNN\Testing"
class_labels = ['Glioma_Tumor', 'Meningioma_Tumor', 'No_Tumor', 'Pituitary_Tumor']

img_paths = []
for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(folder_path, img_file))

random_images = random.sample(img_paths, 12)


plt.figure(figsize=(15, 10))
for idx, img_path in enumerate(random_images):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)
    predicted_label = class_labels[np.argmax(prediction)]

    plt.subplot(3, 4, idx + 1)
    plt.imshow(img)
    plt.title(f"Predicted:\n{predicted_label}", fontsize=10)
    plt.axis('off')

plt.tight_layout()


output_path = "predicted_collage.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved image collage to: {output_path}")


try:
    os.startfile(output_path)
except Exception as e:
    print(f" Couldn't open image: {e}")

print("Script completed.")


