import streamlit as st
import cv2 as cv
import tempfile

#  pip install streamlit opencv-python-headless
import streamlit as st
import cv2 as cv
import numpy as np

# Define the path to the face detection model and its configuration
model_file = "/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "/deploy.prototxt.txt"

# Load the model
net = cv.dnn.readNetFromCaffe(config_file, model_file)


def detect_faces(image):
    """
    Detects faces in an image using OpenCV's DNN module.
    """
    h, w = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv.putText(image, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    return image


def main():
    st.title("Face Detection in Video")

    # File uploader allows users to add their own video
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "asf", "m4v"])

    if uploaded_file is not None:
        # Convert the file to an opencv video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Detect faces
            frame = detect_faces(frame)

            # Display the frame
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            stframe.image(frame)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()


if __name__ == "__main__":
    main()
