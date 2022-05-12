import gc
import os

import cv2
import dlib
import numpy as np
from imutils import face_utils
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    # Vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # The EAR Equation
    EAR = (A + B) / (2.0 * C)
    return EAR


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])

    MAR = (A + B + C) / 3.0
    return MAR


l = []

# Change path to directory where the downloaded ultradd dataset is present
src_dir = "/Users/divy/Downloads/Drowsiness dataset"

for e in os.listdir(src_dir):
    if e == ".DS_Store":
        continue
    e = os.path.join(src_dir, e)
    for f in os.listdir(e):
        if f == ".DS_Store":
            continue
        f = os.path.join(e, f)
        for g in os.listdir(f):
            if g == ".DS_Store":
                continue
            l.append(os.path.join(f, g))

# Required for img_to_finalimg()
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Grab the indexes of the facial landamarks for the left and right eye respectively
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def img_to_finalimg(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # mask = np.zeros((height, width), np.uint8)
    # final_mask = np.zeros((height, width, 3), np.uint8)
    rects = detector(gray, 0)
    left_ear, right_ear, mouth_ear = [], [], []
    for (i, rect) in enumerate(rects):
        try:
            final_mask = np.zeros((height, width, 3), np.uint8)
            mask = np.zeros((height, width), np.uint8)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend]
            mouth = shape[mstart:mend]
            left_ear.append(eye_aspect_ratio(leftEye))
            right_ear.append(eye_aspect_ratio(rightEye))
            mouth_ear.append(mouth_aspect_ratio(mouth))
            # leftEye = cv2.minEnclosingCircle(leftEye)
            # rightEye = cv2.minEnclosingCircle(rightEye)
            # mouth = cv2.fitEllipseDirect(mouth)
            # mask = cv2.circle(mask, (int(leftEye[0][0]), int(leftEye[0][1])), int(leftEye[1]), (255, 255, 255),
            #                   thickness=-1)
            # mask = cv2.circle(mask, (int(rightEye[0][0]), int(rightEye[0][1])), int(rightEye[1]), (255, 255, 255),
            #                   thickness=-1)
            # mask = cv2.ellipse(mask, (int(mouth[0][0]), int(mouth[0][1])), (int(mouth[1][1]), int(mouth[1][0])),
            #                    0, 0, 360, (255, 255, 255), thickness=-1)
            # for i in range(3):
            #     final_mask[:, :, i] = mask
            # final_mask = cv2.bitwise_and(image, final_mask)
            # x, y, w, h = cv2.boundingRect(shape)
            # final_images.append(final_mask[y:y + h, x:x + w])
        except Exception as e:
            print(e)
    return left_ear, right_ear, mouth_ear


# total = 0
# for path in l:
#     cap = cv2.VideoCapture(path)
#     c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     total += c
#     del cap
#     del c
#     gc.collect()
# n = 60000
# sr = int(total / total)
# del total
# gc.collect()
sr = 1
#
# # Change path to the folder where you want to extract images from the video
dest_dir = "Reinforcment_learning/data_csv"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
output_file = os.path.join(dest_dir, "combined_data.csv")
output_file_ = os.path.join(dest_dir, "video_data.csv")
# li = [0, 5, 10]
# for e in li:
#     if not os.path.exists(os.path.join(dest_dir, str(e))):
#         os.mkdir(os.path.join(dest_dir, str(e)))
#
errors = []
file = open(output_file, mode='a')
file.write("lEAR, rEAR, MAR, class\n")
file.close()

file_ = open(output_file_, mode='a')
file_.write("lEAR, rEAR, MAR, class\n")
file_.close()

for path in tqdm(l):
    cap = cv2.VideoCapture(path)
    c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fn = 0

    state = int(os.path.basename(path).split(".")[0])
    # pf_folder = os.path.dirname(path)
    # p_folder = os.path.basename(os.path.dirname(pf_folder))
    #
    # dest_dir = os.path.join(dest_dir, img_folder)
    while fn < c:
        file = open(output_file, mode='a')
        file_ = open(output_file_, mode='a')
        cap.set(1, fn)
        image = cap.read()[1]
        if str(type(image)) != "<class 'numpy.ndarray'>":
            fn = fn + sr
            continue
        left_ear, right_ear, mouth_ear = img_to_finalimg(image)
        csv = ''
        for i in range(len(left_ear)):
            csv += f'{left_ear[i]}, {right_ear[i]}, {mouth_ear[i]}, {state}\n'
        file.write(csv)
        file.close()
        file_.write(csv)
        file_.close()
        del image
        del left_ear
        del right_ear
        del mouth_ear
        del file
        del file_
        del csv
        gc.collect()

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        fn = fn + sr

    cap.release()
    del cap
    del c
    del fn
    gc.collect()

# Take images From drive images
# Change directory to directory where data from the drive is downloaded
# (from https://drive.google.com/drive/folders/1A5hJ2yVEXa6mWMSQzAg0u8_wY8B2OytH)
src_dir = "/Users/divy/Downloads/drive-download-20211001T181747Z-001"

# Change path to the folder where you want to extract images from the video
# dest_dir = "/Users/divy/Downloads/images"
# if not os.path.exists(dest_dir):
#     os.mkdir(dest_dir)

# li = [0, 5, 10]
# for e in li:
#     if not os.path.exists(os.path.join(dest_dir, str(e))):
#         os.mkdir(os.path.join(dest_dir, str(e)))

errors = []
a = 0
for l in os.listdir(src_dir):
    if l == "Eyeclose":
        state = int('10')
    elif l == "Open and no yawn":
        state = int('0')
    elif l == "Yawn":
        state = int('10')
    else:
        continue
    l = os.path.join(src_dir, l)
    file = open(output_file, mode='a')
    csv = ''
    for f in os.listdir(l):
        if f == ".DS_Store":
            continue
        f = os.path.join(l, f)
        image = cv2.imread(f)
        left_ear, right_ear, mouth_ear = img_to_finalimg(image)
        for i in range(len(left_ear)):
            try:
                csv += f'{left_ear[i]}, {right_ear[i]}, {mouth_ear[i]}, {state}\n'
            except:
                print(f)
                errors.append([f])
    file.write(csv)
    file.close()
