import numpy as np
import cv2 as cv

model_bin = "/MobileNetSSD_deploy.caffemodel"
config_test = "/MobileNetSSD_deploy.prototxt.txt"

net = cv.dnn.readNetFromCaffe(config_test, model_bin)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
cap = cv.VideoCapture("/children.mp4")

objName = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

while True:
    ret, image = cap.read()
    image = cv.flip(image, 1)
    if ret is False:
        break
    h, w = image.shape[:2]
    blobImage = cv.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False, crop=False)
    net.setInput(blobImage)
    cvOut = net.forward()

    t, _ = net.getPerfProfile()
    fps = 1000 / (t * 1000.0 / cv.getTickFrequency())
    label = '{:.2f} FPS'.format(fps)
    cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        objIndex = int(detection[1])
        if score > 0.5:
            left = detection[3] * w
            top = detection[4] * h
            right = detection[5] * w
            bottom = detection[6] * h
            cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            label = "score:%.2f, %s" % (score, objName[objIndex])
            cv.putText(image, label, (int(left) - 10, int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8)

    cv.imshow("face-detection-demo", image)
    c = cv.waitKey(10)
    if c == 27:  # ESC tuşuna basıldığında döngüden çık
        break

cv.destroyAllWindows()
