import cv2

# Load pre-trained face recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize id counter
id = 0

# Names related to ids
names = ['None', 'Pouria']

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture video frame.")
        break

    img = cv2.flip(img, 1)  # Flip vertically

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. Preprocessing: Histogram Equalization and Denoising ---
    gray = cv2.equalizeHist(gray)  # Improve contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Remove noise


    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recognize the face
        face_region = gray[y:y + h, x:x + w]
        id, confidence = recognizer.predict(face_region)
        confidence = min(100, confidence)

        if confidence < 100:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        # Display the ID and confidence
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Show processed frames
    cv2.imshow('Camera - Processed', img)  # Processed frame

    k = cv2.waitKey(10) & 0xFF  # Press 'ESC' to exit
    if k == 27:
        break

# Cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()