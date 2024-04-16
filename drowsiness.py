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




def dis(d, f):
    distance = ((d.x - f.x) ** 2 + (d.y - f.y) ** 2) ** 0.5
    return distance




df = pd.read_csv('Drowsy.csv')
x = df.drop("Label", axis=1)
y = df["Label"]
x = np.array(x)
y = np.array(y)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

scaler = StandardScaler()
scaler.fit(x_train)


# print(max(x_train))

x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)


def EAR_pipeline2(image):
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
    # MAR Nodes
    node_160 = detection_result.face_landmarks[0][160]
    node_78 = detection_result.face_landmarks[0][78]
    node_308 = detection_result.face_landmarks[0][308]
    node_303 = detection_result.face_landmarks[0][303]
    node_404 = detection_result.face_landmarks[0][404]
    node_16 = detection_result.face_landmarks[0][16]
    node_11 = detection_result.face_landmarks[0][11]
    node_73 = detection_result.face_landmarks[0][73]
    node_180 = detection_result.face_landmarks[0][180]
    p73180 = dis(node_73, node_180)
    p78308 = dis(node_78, node_308)
    p1116 = dis(node_11, node_16)
    p303404 = dis(node_303, node_404)
    MAR = (p73180 + p1116 + p303404) / (2 * p78308)
    p26l = dis(node_144, node_160)
    p35l = dis(node_158, node_153)
    p14l = dis(node_33, node_133)
    p26r = dis(node_385, node_380)
    p35r = dis(node_373, node_387)
    p14r = dis(node_362, node_263)
    left_EAR = (p26l + p35l) / (2 * p14l)
    right_EAR = (p26r + p35r) / (2 * p14r)
    a = [[left_EAR, right_EAR, MAR]]
    a = np.array(a)
    # a = np.array([a])
    # # scaler = StandardScaler()
    # # a = scaler.fit_transform([a])
    a = scaler.transform(a)
    return a




model = KNeighborsClassifier(n_neighbors=9)

# model = svm.SVC(kernel = 'rbf')

# model = DTC(random_state=10)


font = cv2.FONT_HERSHEY_SIMPLEX

        # org
org = (50, 50)

        # fontScale
fontScale = 1

        # Blue color in BGR
color = (0, 0, 255)

        # Line thickness of 2 px
thickness = 2


model.fit(x_train, y_train)
i = 0
for n in x:
    print(x[i])
    i += 1
n = model.predict(x_test)
a = accuracy_score(y_test, n)
print(a)
accumulator = 0
i = 0
while True:
    webcam = cv2.VideoCapture(1)
    # instead of 0 if we give a video directory it still works
    # 0 is default webcam
    while True:
        successful_frame_read, frame = webcam.read()
        try:
          prediction = model.predict(EAR_pipeline2(frame))
          if prediction == [0]:
            # text = 'Awake'
            accumulator -= 1
          elif prediction == [1]:
            # text = 'Drowsy'
            accumulator += 1
        except:
            text = "No Face Detected"
            image = cv2.putText(frame, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        # Reading an image in default mode
        if accumulator == 100:
            accumulator = 0
            # text = "Drowsy"
            i = 1
        else:
            text = ""
        if accumulator == -100:
            accumulator = 0
            # text = "Awake"
            i = -1
        else:
            text = ""
        if i == 1:
            text = "Drowsy"
        elif i == -1:
            text = "Awake"
        print(accumulator)
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
        image = cv2.putText(frame, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Drowsiness Detection', image)
      #  key = cv2.waitKey(1) 
        # # 1 means that there's 1 millisecond time delay between each frame
        # if keyboard.is_pressed('alt+f4'):
        #     break
        if cv2.waitKey(1) and 0xFF == 'q':
           break

    # get face region coordinates
    # faces = face_cascade.detectMultiScale(gray)
    # get face bounding box for overlay