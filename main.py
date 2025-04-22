import cv2
import requests
from time import time
face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")

base_url = None
add_face_url = base_url
recognize_face_url = base_url

user_to_delete = None
del_time_stamp = 0

def get_cropped_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    return faces

def capture_faces(max_snaps=60):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    name = input("Please enter the name of the person: ")
    face_images = []
    snap_count = 0

    print("📷 Capturing faces. Press 'q' to quit early.")
    cv2.namedWindow("Face Capture", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Face Capture", cv2.WND_PROP_TOPMOST, 1)

    while snap_count < max_snaps:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        faces = get_cropped_faces(frame)

        if len(faces) >= 2:
            print("Please be only one person in front of the camera.")
        elif len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = frame[y:y + h, x:x + w]

            success, encoded_image = cv2.imencode('.jpg', roi)
            if success:
                face_images.append(('images', ('face.jpg', encoded_image.tobytes(), 'image/jpeg')))
                snap_count += 1
                print(f"[{snap_count}/{max_snaps}] Face captured.")

        cv2.putText(frame, f"Capturing face {snap_count}/{max_snaps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quit early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if snap_count == 0:
        print("No faces detected. Exiting.")
        return [], None

    response = requests.post(
        add_face_url,
        files=face_images,
        data={"name": name, "type": "upload"}
    )

    if response.status_code == 200:
        print(f"Response from server: {response.json()}")
    else:
        print(f"Failed to upload faces. Status Code: {response.status_code} ", response.text)

    print(f"Captured {snap_count} faces for {name}.")
    return face_images, name

def recognize_faces():
    global user_to_delete, del_time_stamp
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cv2.namedWindow("Webcam Video", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Webcam Video", cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # print(frame.shape)
        faces = get_cropped_faces(frame)
        face_arr = []
        face_coords = []

        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            success, encoded_image = cv2.imencode('.jpg', roi)
            if success:
                face_arr.append(('images', ('face.jpg', encoded_image.tobytes(), 'image/jpeg')))
                face_coords.append((x, y, w, h))

        if face_arr:
            try:
                response = requests.post(recognize_face_url, files=face_arr, data={"type": "recognize"})
                response.raise_for_status()
                result = response.json()
                print(result)
                recognized_faces = result.get("recognized_faces", [])

                for i, (x, y, w, h) in enumerate(face_coords):
                    if i < len(recognized_faces):
                        face_data = recognized_faces[i]
                        name = face_data.get("name", "Unknown")
                        score = face_data.get("similarity_score", 0.0)
                        if score > 0.50:
                            display_name = name
                            print(f"Member {name} recognized")
                            if user_to_delete:
                                res = requests.post(base_url, data= {"type":"delete","name":user_to_delete})
                                print(f"deleted user {user_to_delete}")
                                user_to_delete = None
                        else: display_name = "Unknown"
                        
                    else:
                        display_name = "Unknown"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.8, (0, 255, 0), 2, cv2.LINE_AA)

            except requests.exceptions.RequestException as e:
                print("Error contacting recognition server:", e)

        cv2.imshow('Webcam Video', frame)

        if user_to_delete and time()-del_time_stamp>45:
            print("Could not authenticate user. Timeout occured")
            user_to_delete = None


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("⏸ Pausing to capture new face...")
            cap.release()
            cv2.destroyWindow("Webcam Video")

            capture_faces()  # Temporarily capture face

            print("🔄 Resuming recognition...")
            cap = cv2.VideoCapture(0)
            cv2.namedWindow("Webcam Video", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Webcam Video", cv2.WND_PROP_TOPMOST, 1)

        elif key == ord('d'):
            user_to_delete = input("Enter the user name which is to be deleted-")
            del_time_stamp = time()
            
        elif key == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with open("server_url.txt","r") as f:
        base_url = f.readline()
    add_face_url = base_url
    recognize_face_url = base_url
    while True:
        try:
            recognize_faces()
        except Exception as e:
            print("Error:", e)
            continue
