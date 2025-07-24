import cv2
import time
from random import randrange

# Time variables for FPS
cTime = 0
pTime = 0

# Open webcam
cam = cv2.VideoCapture(0)

# Set camera resolution (optional but recommended for consistency)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

# Get actual frame size
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' for AVI
out = cv2.VideoWriter('output_blur.avi', fourcc, 20.0, (frame_width, frame_height))

# Load Haar cascade classifier
trained_face_data = cv2.CascadeClassifier(
    'D:\\ML projects\\Face Blur Using Harcascade\\Face Detection Using HaarCascade\\haarcascade_frontalface_default.xml'
)

while True:
    success, frame = cam.read()
    if not success:
        print("Failed to capture frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(gray)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Apply blur to detected faces
    for (x, y, w, h) in face_coordinates:
        # Ensure coordinates are within bounds
        x, y = max(x, 0), max(y, 0)

        face_crop = frame[y:y + h, x:x + w]
        face_blur = cv2.blur(face_crop, (35, 35))
        frame[y:y + h, x:x + w] = face_blur

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (randrange(256), randrange(256), randrange(256)), 2)

    # Show the frame with blur
    cv2.imshow("Blurred Faces", frame)

    # Write the frame to output video
    out.write(frame)

    # Break on Enter key
    key = cv2.waitKey(1)
    if key == 13 or key == 10:  # Enter or carriage return
        break

# Release resources
cam.release()
out.release()
cv2.destroyAllWindows()
