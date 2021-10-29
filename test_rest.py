import requests 
import cv2

content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread("/hdd/tuannm82/mlops-demo/test1.jpg") 

r = requests.post("http://localhost:5001/predict", json={"data":{"ndarray":img.tolist()}})

print(r.content)