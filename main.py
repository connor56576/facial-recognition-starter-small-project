import cv2
import mediapipe as mp

face_detection = mp.solutions.face_detection.FaceDetection() 

video = cv2.VideoCapture(0)

while True:
    success, frame = video.read() # Read a frame from the webcam

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB format for MediaPipe processing

    results = face_detection.process(rgb) # Process the frame to detect faces

    if results.detections:
        for detection in results.detections: # Loop through each detected face

            bbox = detection.location_data.relative_bounding_box # Get the bounding box of the detected face

            h, w, _ = frame.shape # Get the height and width of the frame

            x = int(bbox.xmin * w) 
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            ) # Draw a rectangle around the detected face

    cv2.imshow("Face Tracking", frame) # Display the frame with detected faces

    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("Face Tracking", cv2.WND_PROP_VISIBLE) < 1: # Exit if 'q' is pressed or the window is closed
        break

video.release()
cv2.destroyAllWindows()
#checking if github works