import cv2
from torch_face import faceDetectionRecognition


persons_dir = 'data/persons'
faces_dir = 'data/aligned'
encode_dir = 'data/data.pt'
img = 'data/multiface.jpg'

fdr = faceDetectionRecognition(persons_dir, faces_dir, encode_dir)
encoding_dict = fdr.build_face_storage()

results = fdr.predict(img, encoding_dict)
# results.print()
# results.save()
# results.cropped_face(save=True)

cv2.imshow('Friends', results.show())
cv2.waitKey(0)
cv2.destroyAllWindows()