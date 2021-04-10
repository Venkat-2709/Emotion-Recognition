import os
import sys
import cv2

from classifier import data_gen

# Getting the Command Line Arguments.
LABELS = ['No_pain', 'Pain']
WIDTH = int(sys.argv[1])
HEIGHT = int(sys.argv[2])


# Detects the faces in images and crops the images to the size of WIDTH and HEIGHT.
def detect_face(data_path, output_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for label in LABELS:
        img_path = os.path.join(data_path, label)
        out_path = os.path.join(output_path, label)
        images = os.listdir(img_path)
        for img in images:
            image = cv2.imread(img_path + '/' + img)
            image = cv2.resize(image, (WIDTH + 100, HEIGHT + 100))
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hist_img = cv2.equalizeHist(gray_img)
            face = face_cascade.detectMultiScale(hist_img, scaleFactor=1.2, minNeighbors=3, minSize=(WIDTH, HEIGHT))
            for (x, y, w, h) in face:
                face = image[y:y + HEIGHT, x:x + WIDTH]
                cv2.imwrite(out_path + '/' + img, face)

    pass


# Main Function.
if __name__ == '__main__':
    data_dir = os.path.normpath(sys.argv[3])
    cur_dir = os.getcwd()
    data_path = os.path.join(cur_dir, data_dir)
    data = os.listdir(data_path)
    output_path = os.path.join(cur_dir, 'OutputData/' + str(WIDTH))

    data_dict = dict()

    for set_ in data:
        data_folder = os.path.join(data_path, set_)
        output_folder = os.path.join(output_path, set_)
        detect_face(data_folder, output_folder)
        data_dict[str(set_)] = output_folder

    data_gen(data_dict)
