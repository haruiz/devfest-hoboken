import matplotlib.pylab as plt
import numpy as np
import urllib.request
from PIL import Image as PILImage
from urllib.error import HTTPError
import requests


def url2img(url):
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        image = PILImage.open(resp).convert("RGB")
        image = image.resize((180, 180))
        image = np.asarray(image)
        return image
    except HTTPError as err:
        if err.code == 404:
            raise Exception("Image not found")
        elif err.code in [403, 406]:
            raise Exception("Forbidden image, it can not be reached")
        else:
            raise


def send_image_to_server(*images_uris):
    images = list(map(url2img, images_uris))
    images_batch = np.stack(images, axis=0)

    # Send the request to the server
    response = requests.post("http://localhost:8501/v1/models/flowers:predict",
                                json={"instances": images_batch.tolist()})

    response = response.json()
    labels = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
    for img_url, img_p in zip(images_uris, response["predictions"]):
        label_idx = np.argmax(img_p)
        print(f"{img_url} -> Predicted label: {labels[label_idx]}")



if __name__ == '__main__':
    for i in range(1, 100):
        send_image_to_server("https://upload.wikimedia.org/wikipedia/commons/e/ea/Tulipan_%28Ama%29.JPG",
                             "https://live.staticflickr.com/3443/3218530065_064d10b5db_b.jpg")

    # url = "https://upload.wikimedia.org/wikipedia/commons/e/ea/Tulipan_%28Ama%29.JPG"
    # img1 = url2img(url)
    #
    # url = "https://live.staticflickr.com/3443/3218530065_064d10b5db_b.jpg"
    # img2 = url2img(url)
    #
    # ax = plt.subplot(1, 2, 1)
    # ax.imshow(img1)
    #
    # ax = plt.subplot(1, 2, 2)
    # ax.imshow(img2)
    #
    # plt.show()
    #
    # batch = np.stack([img1, img2])
    #
    # print(batch.shape)
    #
    # print(batch.tolist())

    # img = np.expand_dims(img, axis=0)
    # print(img.shape)
    # print(img.tolist())
    # plt.imshow(img[0])
    # plt.show()