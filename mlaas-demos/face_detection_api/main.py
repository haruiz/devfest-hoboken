
from PIL import Image as PILImage
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import vision
from google.oauth2 import service_account
import os

class ImageReader:
    @staticmethod
    def read_image_from_url(url):
        response = requests.get(url)
        image = PILImage.open(BytesIO(response.content))
        return image
    
    @staticmethod
    def read_image_from_file(file):
        image = PILImage.open(file)
        return image
    
    @classmethod
    def read_image(cls, image_path):
        """
        Read image from file or url
        """
        if image_path.startswith("http"):
            return cls.read_image_from_url(image_path)
        else:
            return cls.read_image_from_file(image_path)
        
        
if __name__ == "__main__":

    # read image
    image_path = "https://imageio.forbes.com/specials-images/dam/imageserve/1084793354/960x0.jpg?format=jpg&width=960"
    image_path = "https://goodlifefamilymag.com/wp-content/uploads/2018/11/happy.jpg"
    image = ImageReader.read_image(image_path)
    plt.imshow(image)
    plt.show()

    # authenicate to google cloud
    service_account_key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service-account-key.json")
    credentials = service_account.Credentials.from_service_account_file(service_account_key_path)
    vision_api_client = vision.ImageAnnotatorClient(credentials=credentials)

    # convert image to bytes
    with BytesIO() as image_bytes:
        image.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()

    # send request to google cloud vision api
    response = vision_api_client.face_detection(image=vision.Image(content=image_bytes), max_results=10)
    detected_faces = response.face_annotations
    print(response)
    
      
    # draw bounding box
    image_arr = np.array(image)
    faces = []
    for face_detection_output in detected_faces:
        vertices = face_detection_output.bounding_poly.vertices
        # is_happy = face_detection_output.joy_likelihood == vision.Likelihood.VERY_LIKELY
        # if is_happy:
        bounding_box = face_detection_output.bounding_poly
        rect = [
            (bounding_box.vertices[0].x, bounding_box.vertices[0].y),
            (bounding_box.vertices[1].x, bounding_box.vertices[1].y),
            (bounding_box.vertices[2].x, bounding_box.vertices[2].y),
            (bounding_box.vertices[3].x, bounding_box.vertices[3].y),
        ]
        faces.append(image_arr[rect[0][1]:rect[2][1], rect[0][0]:rect[2][0]])

    # draw faces on image
    cols = 4
    rows = len(faces) // cols
    fig= plt.figure()
    for i, happy_face in enumerate(faces):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(happy_face)
    
    plt.tight_layout()
    plt.show()

    
            