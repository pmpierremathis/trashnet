import os

from google.cloud import storage
from termcolor import colored
from trashnet.params import BUCKET_NAME

def storage_upload(image, label, rm = False):
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"user_images/{label}/{image}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(image)
    print(colored(f"=> image uploaded to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    
    #rm = True
    if rm:
        os.remove(image)
        print("file uploaded")


if __name__ ==  "__main__":
    storage_upload("cardboard1.jpg", "cardboard")