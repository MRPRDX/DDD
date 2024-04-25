# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import svm
import cv2 as cv
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import keyboard

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


# STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.png")

# # STEP 4: Detect face landmarks from the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


def dis(d, f, x, y):
    distance = ((d.x - f.x) ** 2 + (d.y - f.y) ** 2) ** 0.5
    return distance


df_EAR = pd.read_csv('EAR.csv')
x_EAR = df_EAR.drop("Label", axis=1)
y_EAR = df_EAR["Label"]
x_EAR = np.array(x_EAR)
y_EAR = np.array(y_EAR)

# transformer = Normalizer().transform(x)
# x = transformer
# transformer.transform(x)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# # print(x)


# try:
#   i = 0
#   while True:
#     x[i][0] += 1
#     x[i][1] += 1
#     x[i][2] += 1
#     i += 1
# except:
#   pass

x_train_EAR, x_test_EAR, y_train_EAR, y_test_EAR = train_test_split(x_EAR, y_EAR, test_size=0.2, random_state=10)

scaler_EAR = StandardScaler()
scaler_EAR.fit(x_train_EAR)

# print(max(x_train))

x_test_EAR = scaler_EAR.transform(x_test_EAR)
x_train_EAR = scaler_EAR.transform(x_train_EAR)

df_MAR = pd.read_csv('Yawn.csv')
x_MAR = df_MAR.drop("Label", axis=1)
y_MAR = df_MAR["Label"]
x_MAR = np.array(x_MAR)
y_MAR = np.array(y_MAR)

x_train_MAR, x_test_MAR, y_train_MAR, y_test_MAR = train_test_split(x_MAR, y_MAR, test_size=0.2, random_state=10)

scaler_MAR = StandardScaler()
scaler_MAR.fit(x_train_MAR)

x_test_MAR = scaler_MAR.transform(x_test_MAR)
x_train_MAR = scaler_MAR.transform(x_train_MAR)

def EAR_nodes(image):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(img)
    # EAR Nodes
    node_160 = detection_result.face_landmarks[0][160]
    node_144 = detection_result.face_landmarks[0][144]
    node_33 = detection_result.face_landmarks[0][33]
    node_133 = detection_result.face_landmarks[0][133]
    node_153 = detection_result.face_landmarks[0][153]
    node_158 = detection_result.face_landmarks[0][158]
    node_385 = detection_result.face_landmarks[0][385]
    node_380 = detection_result.face_landmarks[0][380]
    node_362 = detection_result.face_landmarks[0][362]
    node_263 = detection_result.face_landmarks[0][263]
    node_373 = detection_result.face_landmarks[0][373]
    node_387 = detection_result.face_landmarks[0][387]
    list_1 = [node_144, node_160, node_158, node_153, node_33, node_133, node_385, node_380, node_373, node_387, node_362, node_263]
    return list_1
def EAR(image):
    node = EAR_nodes(image)
    p26l = dis(node[0], node[1], x_EAR, y_EAR)
    p35l = dis(node[2], node[3], x_EAR, y_EAR)
    p14l = dis(node[4], node[5], x_EAR, y_EAR)
    p26r = dis(node[6], node[7], x_EAR, y_EAR)
    p35r = dis(node[8], node[9], x_EAR, y_EAR)
    p14r = dis(node[10], node[11], x_EAR, y_EAR)
    left_EAR = (p26l + p35l) / (2 * p14l)
    right_EAR = (p26r + p35r) / (2 * p14r)
    a = [[left_EAR, right_EAR]]
    a = np.array(a)
    # a = np.array([a])
    # # scaler = StandardScaler()
    # # a = scaler.fit_transform([a])
    a = scaler_EAR.transform(a)
    return a

def MAR_nodes(image):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(img)
    # MAR Nodes
    node_78 = detection_result.face_landmarks[0][78]
    node_308 = detection_result.face_landmarks[0][308]
    node_303 = detection_result.face_landmarks[0][303]
    node_404 = detection_result.face_landmarks[0][404]
    node_16 = detection_result.face_landmarks[0][16]
    node_11 = detection_result.face_landmarks[0][11]
    node_73 = detection_result.face_landmarks[0][73]
    node_180 = detection_result.face_landmarks[0][180]
    list_2 = [node_73, node_180, node_78, node_308, node_11, node_16, node_303, node_404]
    return list_2

def MAR(image):
    node_1 = MAR_nodes(image)
    p73180 = dis(node_1[0], node_1[1], x_MAR, y_MAR)
    p78308 = dis(node_1[2], node_1[3], x_MAR, y_MAR)
    p1116 = dis(node_1[4], node_1[5], x_MAR, y_MAR)
    p303404 = dis(node_1[6], node_1[7], x_MAR, y_MAR)
    MAR = (p73180 + p1116 + p303404) / (2 * p78308)
    b = [[MAR]]
    b = np.array(b)
    # a = np.array([a])
    # # scaler = StandardScaler()
    # # a = scaler.fit_transform([a])
    b = scaler_MAR.transform(b)
    return b


model_EAR = KNeighborsClassifier(n_neighbors=3)

# model = svm.SVC(kernel = 'rbf')

# model = DTC(random_state=10)

model_EAR.fit(x_train_EAR, y_train_EAR)

model_MAR = KNeighborsClassifier(n_neighbors=9)

model_MAR.fit(x_train_MAR, y_train_MAR)
i = 0
for n in x_EAR:
    print(x_EAR[i])
    i += 1
n = model_MAR.predict(x_test_MAR)
a = accuracy_score(y_test_MAR, n)
print(a)
accumulator = 0
i = 0

font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

while True:
    webcam = cv2.VideoCapture(0)
    # webcam.set(3, 1920)
    # webcam.set(4, 1080)
    # instead of 0 if we give a video directory it still works
    # 0 is default webcam
    while True:
        successful_frame_read, frame = webcam.read()
        try:
            prediction_EAR = model_EAR.predict(EAR(frame))
            prediction_MAR = model_MAR.predict(MAR(frame))
            if prediction_EAR == [0] and prediction_MAR == [0]:
                # text = 'Awake'
                accumulator -= 3
            elif prediction_EAR == [1] and prediction_MAR == [1]:
                # text = 'Drowsy'
                accumulator += 3
            elif prediction_EAR == [1] and prediction_MAR == [0]:
                accumulator += 2
            elif prediction_EAR == [0] and prediction_MAR == [1]:
                accumulator -= 2
            if accumulator >= 60:
                accumulator = 0
                # text = "Drowsy"
                i = 1
            else:
                text = ""
            if accumulator <= -50:
                accumulator = 0
                # text = "Awake"
                i = -1
            else:
                text = ""
            if i == 1:
                text = "Drowsy"
            elif i == -1:
                text = "Awake"
            for n in EAR_nodes(frame):
                cv2.circle(frame, (int(n.x * 640), int(n.y * 480)), 1, (0, 0, 255), -1)
            for m in MAR_nodes(frame):
                cv2.circle(frame, (int(m.x*640), int(m.y*480)), 1, (0, 0, 255), -1)
            # for y in range(11):
            #     cv2.circle(frame, MAR_nodes(frame)[y], 1,(0,0,255))
        except:
            text = "No Face Detected"
            # image = cv2.putText(frame, text, org, font,
            #                     fontScale, color, thickness, cv2.LINE_AA)
        # Reading an image in default mode
        # print(accumulator)
        image = cv2.imread
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 0, 255)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        print(accumulator)
        image = cv2.putText(frame, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Drowsiness Detection', image)
        #  key = cv2.waitKey(1)
        # # 1 means that there's 1 millisecond time delay between each frame
        # if keyboard.is_pressed('alt+f4'):
        #     break
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    # get face region coordinates
    # faces = face_cascade.detectMultiScale(gray)
    # get face bounding box for overlay
