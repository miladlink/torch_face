import cv2
from torch_face import faceDetectionRecognition


persons_dir = 'data/persons'
faces_dir = 'data/aligned'
encode_dir = 'data/data.pt'

fdr = faceDetectionRecognition(persons_dir, faces_dir, encode_dir)
encoding_dict = fdr.build_face_storage()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    results = fdr.predict(img, encoding_dict, landmarks=True)
    cv2.imshow('face detector', results.show())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break