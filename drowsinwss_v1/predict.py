import cv2
import dlib
import numpy as np
from imutils import face_utils
from skimage.transform import resize
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

N_CLASSES = 3  # CHANGE HERE, total number of classes
IMG_HEIGHT = 512  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 512  # CHANGE HERE, the image width to be resized to
CHANNELS = 3


# def build_model(num_classes):
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, IMG_HEIGHT, IMG_WIDTH)
#     else:
#         input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
#
#     model = Sequential()
#     model.add(Conv2D(16, (3, 3), input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.15))
#     model.add(Conv2D(32, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(64))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['accuracy'])
#     return model


def build_model(num_classes):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_HEIGHT, IMG_WIDTH)
    else:
        input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

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
    rects = detector(gray, 0)
    # final_mask = np.zeros((height, width, 3), np.uint8)
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
            # leftEye = cv2.fitEllipseDirect(leftEye)
            # rightEye = cv2.fitEllipseDirect(rightEye)
            mouth = cv2.fitEllipseDirect(mouth)
            mask = cv2.circle(mask, (int(leftEye[0][0]), int(leftEye[0][1])), int(leftEye[1]), (255, 255, 255),
                              thickness=-1)
            mask = cv2.circle(mask, (int(rightEye[0][0]), int(rightEye[0][1])), int(rightEye[1]), (255, 255, 255),
                              thickness=-1)
            # mask = cv2.ellipse(mask, (int(leftEye[0][0]), int(leftEye[0][1])), (int(leftEye[1][1]), int(leftEye[1][0])),
            #                    0, 0, 360, (255, 255, 255),thickness=-1)
            # mask = cv2.ellipse(mask, (int(rightEye[0][0]), int(rightEye[0][1])), (int(rightEye[1][1]), int(rightEye[1][0])),
            #                    0, 0, 360, (255, 255, 255), thickness=-1)
            mask = cv2.ellipse(mask, (int(mouth[0][0]), int(mouth[0][1])), (int(mouth[1][1]), int(mouth[1][0])),
                               0, 0, 360, (255, 255, 255), thickness=-1)
            for j in range(3):
                final_mask[:, :, j] = mask
            final_mask = cv2.bitwise_and(image, final_mask)
            x, y, w, h = cv2.boundingRect(shape)
            final_images.append([final_mask[y:y + h, x:x + w], (x, y, w, h)])
        except Exception as e:
            print(e)
    return final_images


def predict(path, type):
    if path == "0":
        path = int(path)
    cap = cv2.VideoCapture(path)
    fn = 0
    if type == "2":
        model = build_model(N_CLASSES-1)
        model.load_weights('binary_classes/model_binary.h5')
    elif type == "3":
        model = build_model(N_CLASSES)
        model.load_weights('all_classes/model_all.h5')
    else:
        print("Enter valid type")
    while True:
        cap.set(1, fn)
        ret, image = cap.read()
        if ret:
            lst_images = img_to_finalimg(image)
            for i in range(len(lst_images)):
                try:
                    img = lst_images[i][0]
                    # cv2.imshow(f'Video{i}', img)
                    x, y, w, h = lst_images[i][1]
                    img = resize(img, (512, 512)) * 255

                    img = img.astype(np.uint8)
                    img = np.expand_dims(img, axis=0)
                    preds = np.argmax(model.predict(img))
                    if preds == 0:
                        text = "alert"
                        color = (0, 255, 0)
                    elif preds == 1:
                        text = "drowsy"
                        color = (0, 0, 255)
                    elif preds == 2:
                        text = "low vigilant"
                        color = (0, 165, 255)
                    cv2.putText(image, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                except Exception as e:
                    print(e)

            # Display the resulting image
            cv2.imshow('Video', image)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_type = input("Type 2 for predicting drowsy and alert and 3 to include low vigilant:- ")
    user_input = input("Pls enter the path of video which you want to predict(0 for using webcam):- ")
    predict(user_input, model_type)
