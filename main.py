import cv2
import requests

# Load the pre-trained classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")

base_url = "https://superb-bbs-contents-invoice.trycloudflare.com/"




add_face_url = base_url   # Updated add_face_url for proper endpoint
recognize_face_url = base_url 

def get_cropped_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    return faces

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def capture_faces(max_snaps=20):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    name = input("Please enter the name of the person: ")
    face_images = []  # Array to store the captured face images
    snap_count = 0

    print("📷 Capturing faces. Press 'q' to quit early.")

    while snap_count < max_snaps:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Get detected faces
        faces = get_cropped_faces(frame)
        
        # Draw rectangles around detected faces
        if len(faces) >= 2:
            print("Please be only one person in front of the camera.")
        elif len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frame[y:y + h, x:x + w]
            success, encoded_image = cv2.imencode('.jpg', roi)
            if success:
                # Convert to bytes and append to the files list
                face_images.append(('images', ('face.jpg', encoded_image.tobytes(), 'image/jpeg')))
            
            snap_count += 1
            print(f"[{snap_count}/{max_snaps}] Face captured.")

        # Show the frame with progress
        cv2.putText(frame, f"Capturing face {snap_count}/{max_snaps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Capture", frame)

        # Wait for 'q' to quit or continue to next image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quit early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if snap_count == 0:
        print("No faces detected. Exiting.")
        return [], None

    # Send images along with the person's name to the server
    # print(face_images)
    response = requests.post(
        add_face_url,
        files=face_images,
        data={"name":name, "type":"upload"}
        # 👈 Send the name along with the images
    )

    if response.status_code == 200:
        print(f"Response from server: {response.json()}")
    else:
        print(f"Failed to upload faces. Status Code: {response.status_code} ",response.text)

    # Ask for the person's name after capturing images


    print(f"Captured {snap_count} faces for {name}.")
    
    # Return the face images along with the name
    return face_images, name


def recognize_faces():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame")
            break

        # Get the faces detected
        faces = get_cropped_faces(frame)

        # Draw bounding boxes around the faces
        face_arr = []
        for (x, y, w, h) in faces:
            # Draw a rectangle around each detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frame[y:y + h, x:x + w]
            success, encoded_image = cv2.imencode('.jpg', roi)
            if success:
                # Convert to bytes and append to the files list
                face_arr.append(('images', ('face.jpg', encoded_image.tobytes(), 'image/jpeg')))
            
        # Make a POST request to recognize faces
        if face_arr:
            response = requests.post(recognize_face_url, files=face_arr, data={"type":"recognize"})
            print(f"Response from recognition server: {response.text}")
            try:
                print(response.json())  # This shows the {'status': ..., 'message': ...}
            except Exception:
                print(response.text)  # 

        # Display the resulting frame with bounding boxes
        cv2.imshow('Webcam Video', frame)

        # Press 'q' to quit the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Main flow
if __name__ == "__main__":
    inp = input("Enter 1 for recognizing or 2 for adding a person: ")
    if inp == "1":
        recognize_faces()
    elif inp == "2":
        captured_faces, person_name = capture_faces(max_snaps=20)
