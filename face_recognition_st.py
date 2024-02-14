import streamlit as st
import cv2 as cv
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.net = cv.dnn.readNetFromCaffe(
            "/MobileNetSSD_deploy.prototxt.txt",
            "/MobileNetSSD_deploy.caffemodel"
        )
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.objName = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        h, w = img.shape[:2]
        blobImage = cv.dnn.blobFromImage(img, 0.007843, (300, 300), (127.5, 127.5, 127.5), False, crop=False)
        self.net.setInput(blobImage)
        cvOut = self.net.forward()

        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            objIndex = int(detection[1])
            if score > 0.5:
                left = detection[3] * w
                top = detection[4] * h
                right = detection[5] * w
                bottom = detection[6] * h
                cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
                label = "score:%.2f, %s" % (score, self.objName[objIndex])
                cv.putText(img, label, (int(left) - 10, int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8)

        return img


# Install streamlit-webrtc via pip if you haven't already
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
except ImportError:
    st.write(
        "Error: streamlit-webrtc is not installed. Please run `pip install streamlit-webrtc` to install the package.")
    raise

st.title("Video Object Detection")

# Using WebRTC to process video stream
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
