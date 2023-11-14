# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
#
# # Load the saved model
# model_path = "scalp_inspector.h5"
# model = tf.keras.models.load_model(model_path)
#
#
# # Function to preprocess the input image
# def preprocess_image(image_path, image_size=256):
#     # Read the image
#     image = cv2.imread(image_path)
#     # Convert the image from BGR to RGB format
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
#     # Resize the image to the specified size
#     image = tf.image.resize(image, (image_size, image_size))
#     # Convert the pixel values to floating-point and normalize to [0, 1]
#     image = image.numpy().astype("uint8")
#     # Add a batch dimension to the image
#     image = np.expand_dims(image, 0)
#     return image
#
#
# # Path to the test image you want to predict
# test_image_path = "alopecia.jpg"
#
# # Preprocess the test image
# input_image = preprocess_image(test_image_path)
#
# # Save the input image as a JPEG
# image_to_save = Image.fromarray((input_image[0] * 255).astype(np.uint8))
# image_to_save.save("input_image.jpg", "JPEG")
#
# # Make predictions on the test image
# predictions = model.predict(input_image)
#
# print(predictions)
# # Get the class index with the highest probability
# predicted_class_index = np.argmax(predictions[0])
#
# # Assuming you have a list of class names, get the predicted class name
# class_names = ['alopecia', 'dandruff', 'folliculitis', 'healthy', 'seborrheic']
# # Replace with your actual class names
#
# # Print the predicted class name
# predicted_class_name = class_names[predicted_class_index]
# print("Predicted Class:", predicted_class_name)
#
# # Convert the image back to uint8 for displaying
# display_image = (input_image[0] * 255).astype(np.uint8)
#
# # Display the processed image in real-time
# cv2.imshow("Processed Image", display_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
import tempfile
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model("scalp_inspector.h5", compile=False)

# Define the list of class names
class_names = ['alopecia', 'dandruff', 'folliculitis', 'healthy', 'seborrheic']


def preprocess_image(image_path, image_size=256):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image to the specified size
    image = tf.image.resize(image, (image_size, image_size))
    # Convert the pixel values to floating-point and normalize to [0, 1]
    image = image.numpy().astype("uint8")
    # Add a batch dimension to the image
    image = np.expand_dims(image, 0)
    return image


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    # Save the image to a temporary file in binary mode
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode='wb') as temp_image_file:
        temp_image_path = temp_image_file.name
        image_file.save(temp_image_path)

    image = preprocess_image(temp_image_path, 256)
    # Normalize the pixel values to [0, 1]
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])
    print(class_name)
    # Create the response
    response = {
        "class": class_name,
        "confidence_score": confidence_score
    }

    display_image = (image[0] * 255).astype(np.uint8)
    # Display the processed image in real-time
    cv2.imshow("Processed Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
