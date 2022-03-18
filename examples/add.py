from torch_face import faceDetectionRecognition

persons_dir = 'persons'
faces_dir = 'aligned'
encode_dir = 'data.pt'

fdr = faceDetectionRecognition(persons_dir, faces_dir, encode_dir)

fdr.addFaces('new face', 'Milad')