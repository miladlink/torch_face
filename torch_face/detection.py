import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



class Detections:
    """
    Detection and recognition class for inference results
    Keyword Arguments:
        :param img: {Image.Image}
        :param comps: {tuple} -> (predicted_names, cropped_faces)
        :param outputs: {tuple} -> (locations, probs, landmarks if landmarks True)
        :param landmarks: {bool} -> True or False
    """
    def __init__(self, img, comps, outputs, landmarks):
        self.img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.comps = comps
        self.outputs = outputs
        self.landmarks = landmarks

    def display(self, pprint=False, loc=False, crop=False, show=False, plot=False, save=False, save_dir=''):
        if self.comps is not None:
            names, crops = self.comps
            if pprint:
                # showing names and probs of detected face
                print('Names:{}, Probs:{}'.format(names, self.outputs[1]))

            if loc:
                # location of detected faces
                return dict(Boxes=self.outputs[0], landMarks=self.outputs[2] if self.landmarks else None)

            if crop:
                # cropped_face image array and save it if u want
                if save:
                    os.makedirs(f'{save_dir}/crops', exist_ok=True)
                    for i in range(len(crops)):
                        file = f'{save_dir}/crops/{names[i]}.jpg'
                        cv2.imwrite(file, self.to_numpy(crops[i]) * 255)
                        print('Cropped Faces saved!')
                return crops

            if show:
                # draw bounding box names and aligned point on the face
                for i, box in enumerate(self.outputs[0]):
                    x1, y1, x2, y2 = list(map(lambda x: int(x), box))
                    scale = round(((x2 - x1) + 78) / 75)
                    (w, h), _ = cv2.getTextSize(names[i], cv2.FONT_HERSHEY_PLAIN, scale, 2)
                    cv2.rectangle(self.img, (x1, y2 + h + 2), (x1 + w, y2), (255, 0, 0), -1)
                    cv2.rectangle(self.img, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)
                    cv2.putText(self.img, names[i], (x1, y2 + h), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), 2)

                    if self.landmarks:
                        for point in self.outputs[2][i]:
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)
                return self.img

            if plot:
                # plot cropped faces
                faces = self.cropped_face()
                plt.figure(figsize=(15, 15))
                for i in range(len(faces)):
                    plt.subplot(3, 3, i + 1)
                    plt.imshow(faces[i])
                    plt.axis('off')
                    plt.title(names[i])
                plt.show()

            if save:
                # saving draw image
                file = save_dir + '/saved_pred.jpg'
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(file, self.show())
                print('Image Saved!')

        else:
            if pprint or loc or crop or plot or save:
                print('No Face Detected!')
            else:
                return self.img

    def print(self):
        self.display(pprint=True)

    def locations(self):
        return self.display(loc=True)

    def cropped_face(self, save=False, save_dir='results'):
        return self.display(crop=True, save=save, save_dir=save_dir)

    def show(self):
        return self.display(show=True)

    def plot(self):
        return self.display(plot=True)

    def save(self, save_dir='results'):
        self.display(save=True, save_dir=save_dir)

    @staticmethod
    def to_numpy(img):
        return img.numpy().transpose(1, 2, 0)