import gc
import os

import cv2
import dlib
import numpy as np
from imutils import face_utils
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm

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
    final_images = []
    for (i, rect) in enumerate(rects):
        try:
            final_mask = np.zeros((height, width, 3), np.uint8)
            mask = np.zeros((height, width), np.uint8)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend]
            mouth = shape[mstart:mend]
            leftEye = cv2.minEnclosingCircle(leftEye)
            rightEye = cv2.minEnclosingCircle(rightEye)
            mouth = cv2.fitEllipseDirect(mouth)
            mask = cv2.circle(mask, (int(leftEye[0][0]), int(leftEye[0][1])), int(leftEye[1]), (255, 255, 255),
                              thickness=-1)
            mask = cv2.circle(mask, (int(rightEye[0][0]), int(rightEye[0][1])), int(rightEye[1]), (255, 255, 255),
                              thickness=-1)
            mask = cv2.ellipse(mask, (int(mouth[0][0]), int(mouth[0][1])), (int(mouth[1][1]), int(mouth[1][0])),
                               0, 0, 360, (255, 255, 255), thickness=-1)
            for i in range(3):
                final_mask[:, :, i] = mask
            final_mask = cv2.bitwise_and(image, final_mask)
            x, y, w, h = cv2.boundingRect(shape)
            final_images.append(final_mask[y:y + h, x:x + w])
        except Exception as e:
            print(e)
    return final_images


total = 0
for path in l:
    cap = cv2.VideoCapture(path)
    c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total += c
    del cap
    del c
    gc.collect()
n = 60000
sr = int(total / n)
del total
gc.collect()

# Change path to the folder where you want to extract images from the video
dest_dir = "/Users/divy/Downloads/images"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

li = [0, 5, 10]
for e in li:
    if not os.path.exists(os.path.join(dest_dir, str(e))):
        os.mkdir(os.path.join(dest_dir, str(e)))

errors = []
for path in tqdm(l):
    cap = cv2.VideoCapture(path)
    c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fn = 0

    img_folder = os.path.basename(path).split(".")[0]
    pf_folder = os.path.dirname(path)
    p_folder = os.path.basename(os.path.dirname(pf_folder))

    dest_dir = os.path.join(dest_dir, img_folder)
    while fn < c:
        cap.set(1, fn)
        image = cap.read()[1]
        if str(type(image)) != "<class 'numpy.ndarray'>":
            fn = fn + sr
            continue
        final_imgs = img_to_finalimg(image)
        for i in range(len(final_imgs)):
            try:
                image_path = os.path.join(dest_dir, f"{str(len(os.listdir(dest_dir)))}.jpeg")
                final_img = final_imgs[i]
                final_img = resize(final_img, (512, 512)) * 255
                final_img = final_img.astype(np.uint8)
                imsave(image_path, final_img)
                del image_path
                del final_img
                gc.collect()
            except Exception as exp:
                for err in [path, fn, final_imgs, exp]:
                    print(err)
                print()
                errors.append([path, fn, final_imgs, exp])
        del final_imgs
        del image
        gc.collect()

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        fn = fn + sr
    cap.release()
    del cap
    del c
    del fn
    del img_folder
    del pf_folder
    del p_folder
    gc.collect()

# Take images From drive images
# Change directory to directory where data from the drive is downloaded
# (from https://drive.google.com/drive/folders/1A5hJ2yVEXa6mWMSQzAg0u8_wY8B2OytH)
src_dir = "/Users/divy/Downloads/drive-download-20211001T181747Z-001"

# Change path to the folder where you want to extract images from the video
dest_dir = "/Users/divy/Downloads/images"
if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

li = [0, 5, 10]
for e in li:
    if not os.path.exists(os.path.join(dest_dir, str(e))):
        os.mkdir(os.path.join(dest_dir, str(e)))

errors = []
a = 0
for l in os.listdir(src_dir):
    if l == "Eyeclose":
        dest_dir = os.path.join(dest_dir, str(10))
    elif l == "Open and no yawn":
        dest_dir = os.path.join(dest_dir, str(0))
    elif l == "Yawn":
        dest_dir = os.path.join(dest_dir, str(10))
    else:
        continue
    l = os.path.join(src_dir, l)
    for f in os.listdir(l):
        if f == ".DS_Store":
            continue
        f = os.path.join(l, f)
        image = cv2.imread(f)
        final_imgs = img_to_finalimg(image)
        for i in range(len(final_imgs)):
            try:
                image_path = os.path.join(dest_dir, f"{str(len(os.listdir(dest_dir)))}.jpeg")
                final_img = final_imgs[i]
                final_img = resize(final_img, (512, 512)) * 255
                final_img = final_img.astype(np.uint8)
                imsave(image_path, final_img)
                a += 1
            except:
                errors.append([f])
