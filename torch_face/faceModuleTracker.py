import os
import cv2
import shutil
import requests
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch_face.detection import Detections
from facenet_pytorch import MTCNN, InceptionResnetV1


class faceDetectionRecognition:
    """
    face Detection and Recognition Module.

    this class by using facenet-pytorch package first detects and encodes faces who wanted to recognize
    then save encodes in .pt file to future usage (every time you can add new face to data storage of faces)

    After creation of encoded file prediction based on comparison of new image encode and encodes is done

    Keyword Arguments:
        :param person_dir: {str} -> directory of persons who is wanted to recognize (every person has one or more image in a directory)
        :param faces_dir: {str} -> a directory to save aligned images
        :param encode_dir: {str} -> a directory to save or load face encoded data
        :param pretrained: {str} -> 'vggface2' 107Mb or 'casia-webface' 111Mb
    """
    def __init__(self, person_dir, faces_dir, encode_dir=None, pretrained='vggface2'):
        self.person_dir = person_dir
        self.names = os.listdir(self.person_dir)
        self.faces_dir = faces_dir
        self.encode_dir = encode_dir

        self.face_detector = MTCNN(image_size=160, margin=0.1, thresholds=[0.6, 0.7, 0.85], keep_all=True)
        self.face_encoder = InceptionResnetV1(pretrained=pretrained).eval()

    def build_face_storage(self):
        """
        encode persons image and save it to data.pt file

        :return: {dict}
        a dictionary of person names and mean encode of each person {person_name:encode}
        """
        if self.encode_dir is None:
            encoding_dict = {}
            for name in os.listdir(self.person_dir):
                encodes = []
                # images of one person
                for img_path in glob(f'{self.person_dir}/{name}/*'):
                    # save_name for aligned image
                    save_name = img_path.split('/')[-1]
                    encode, img_cropped = self.encoder(img_path, name, save_name)
                    encodes.append(encode)
                    # mean of encodes for one person
                    mean_encode = torch.mean(torch.vstack(encodes), dim=0)
                    encoding_dict[name] = mean_encode

            # saving all of encodes
            torch.save(encoding_dict, 'data.pt')
            print('Face Storage Created!')
            return encoding_dict
        else:
            try:
                encoding_dict = torch.load(self.encode_dir)
                print('Face Storage Loaded!')
                return encoding_dict
            except:
                print('pt file has not valid content')

    def addFaces(self, path, name):
        """
        adding new face encode to encodes
        :param path: {str} -> path of a directory contains new face images
        :param name: {str} -> name of new face
        :return: None
        """
        if name not in self.names:
            # create a directory for new person and copy images to it
            os.mkdir(f'{self.person_dir}/{name}')
            for img_name in os.listdir(path):
                src = os.path.join(path, img_name)
                dst = os.path.join(self.person_dir, name, img_name)
                shutil.copy(src, dst)

            encoding_dict = torch.load(self.encode_dir)
            encodes = []
            for img_path in glob(f'{self.person_dir}/{name}/*'):
                save_name = img_path.split('/')[-1]
                encode, img_cropped = self.encoder(img_path, name, save_name)
                encodes.append(encode)
                mean_encode = torch.mean(torch.vstack(encodes), dim=0)
                encoding_dict[name] = mean_encode
            torch.save(encoding_dict, 'data.pt')
            print(f"The {name}'s face added!")
        else:
            print(f"The {name}'s face exists!")

    def predict(self, img, encoding_dict, landmarks=False):
        """
        predict results based on Inference class
        :param img:
            {str} -> data/multiface.jpg file
            {url} -> http://.../image.jpg url
            {ndarray} -> OpenCV format image
            {Image.Image} -> PIL format image
        :param encoding_dict: {dict} -> encoded faces from self.build_face_storage
        :param landmarks: {bool} -> if True it draws 5 points of landmarks on the face
        :return: an object of Inference class
        """
        # PIL
        if isinstance(img, Image.Image):
            pass

        # file or URL
        elif isinstance(img, str):
            img = Image.open(requests.get(img, stream=True).raw if str(img).startswith('http') else img)

        # OpenCV
        elif type(img) == np.ndarray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        else:
            print('unknown image type')
            exit(-1)

        comps = self.compare(img, encoding_dict)
        outputs = self.face_detector.detect(img, landmarks)
        return Detections(img, comps, outputs, landmarks)

    def compare(self, img, encoding_dict, conf_thresh=250):
        """
        comparison of new image encode and encoding_dict and choose one person
        if it is close
        :param img: {Image.Image} image to comparison
        :param encoding_dict: {dict} a dictionary of names and encodings
        :param conf_thresh: a threshold to separate known and unknown face
        :return:
            predicted name
            cropped face of predicted name
        """
        crops = self.face_detector(img)
        if crops is not None:
            self.face_encoder.classify = True
            encodes = self.face_encoder(crops).detach()
            names = []
            for i in range(len(encodes)):
                encode = encodes[i]
                distances = {}
                for name, embed in encoding_dict.items():
                    # comparison
                    dist = torch.dist(encode, embed).item()
                    distances[name] = dist
                # min of distance if less than conf_thresh
                min_score = min(distances.items(), key=lambda x: x[1])
                name = min(distances, key=lambda k: distances[k]) if min_score[1] < conf_thresh else 'Unknown'
                names.append(name)

            return names, crops

    def encoder(self, img_path, name, save_name):
        """
        encoding one image and save aligned face

        :param img_path: {str}
        :param name: {str} -> face name
        :param save_name: {str}
        :return:
        """
        img = Image.open(img_path)
        img_cropped = self.face_detector(img, save_path=f'{self.faces_dir}/{name}/{save_name}')
        try:
            self.face_encoder.classify = True
            encode = self.face_encoder(img_cropped).detach()
            return encode, img_cropped
        except ValueError:
            print('No Face detected in one of storage Faces; change or remove image from storage')