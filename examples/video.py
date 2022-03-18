import cv2
from torch_face import faceDetectionRecognition


persons_dir = 'data/persons'
faces_dir = 'data/aligned'
encode_dir = 'data/data.pt'
vid_name = 'input.mp4'
out_name = 'results/output.mp4'

fdr = faceDetectionRecognition(persons_dir, faces_dir, encode_dir)
encoding_dict = fdr.build_face_storage()

cap = cv2.VideoCapture(vid_name)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_name, fourcc, fps, (int(w), int(h)))

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    results = fdr.predict(img, encoding_dict, landmarks=True)
    out.write(results.show())
    cv2.imshow('face detector', results.show())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()