import cv2
import requests
from time import time
face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")

base_url = None
add_face_url = base_url
recognize_face_url = base_url


def get_cropped_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    return faces

def capture_faces(max_snaps=60, rec_mode= False):
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    face_images = []
    snap_count = 0

    print("📷 Capturing faces. Press 'q' to quit early.")
    cv2.namedWindow("Face Capture", cv2.WINDOW_NORMAL)

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

        if not rec_mode:
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
        return []
    
    

    
    return face_images

def add_person():
    name = "xyztpr"
    while(not name or name=="xyztpr"):
        name = input("Enter the name of the person- ")
        if name =="q": return

    face_images= capture_faces(60)
    print(f"Captured {len(face_images)} faces for {name}")
    response = requests.post(
        add_face_url,
        files=face_images,
        data={"name": name, "type": "upload"}
    )

    if response.status_code == 200:
        print(f"Response from server: {response.json()}")
    else:
        print(f"Failed to upload faces. Status Code: {response.status_code} ", response.text)

    

def delete_user(timeout = 45):
    name = "xyztpr2"
    while not name  or name == "xyztpr2":
        name = input("Please Enter name of user to delete - ")
        
    print("Needs Authentication before Deleting. Please be in front of camera")
    
    start_time = time()
    while(True):
        
        rec = recognize_faces()

        if rec:
            res = requests.post(base_url, data= {"type":"delete","name":name})
            print(f"deleted user {name}")
            break
    
        if time()-start_time>timeout:
            print("Could not authenticate user. Timeout occured")
            
        
        
def recognize_faces():

    try:
        face_arr = capture_faces(5)

        response = requests.post(recognize_face_url, files=face_arr, data={"type": "recognize"})
        response.raise_for_status()
        result = response.json()
        print(result)
        
        result= result.get("recognized_faces",[{},{}])
        score = result[0].get("max_score", 0.0)
        name = result[0].get("name","None")
        if score > 0.50:
            print(f"Member {name} recognized")
            return True
        return False
    except Exception as e:
        print(e)

            

if __name__ == "__main__":
    with open("server_url.txt","r") as f:
        base_url = f.readline()
    add_face_url = base_url
    recognize_face_url = base_url
    while True:
        print("Enter 1 for recognizing 2 for adding a person and 3 for deleting a person")
        inp = input()
        if inp=="1":
            recognize_faces()
        if inp=="2":
            add_person()
        if inp=="3":
            delete_user()
