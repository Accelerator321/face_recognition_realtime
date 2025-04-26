import cv2
import requests
from time import time, sleep
import aiohttp
import asyncio
import io
import concurrent.futures
import requests


last_time = 0
rate_limit = 1  # example: 1 second between requests

face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")

base_url = None
add_face_url = base_url
recognize_face_url = base_url
curr_window = "Main_window"
cap = None
rate_limit = 1
last_time = 0

import threading
import queue

frame_queue = queue.LifoQueue(maxsize=60)
curr_context = "default"
# Create lock and context
context_lock = threading.Lock()
activity_lock = threading.Lock()

def change_context(new_context: str):
    """
    Safely update the current context using a lock.

    Args:
        new_context (str): The new context to set.
    """
    global curr_context
    with context_lock:
        curr_context = new_context
    # sleep(0.3)

def camera_thread_func():
    global cap, curr_context, frame_queue

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces
        faces = get_cropped_faces(frame)

        # Only allow 1 face to be captured at a time
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            roi = frame[y:y+h, x:x+w]
            success, encoded_image = cv2.imencode('.jpg', roi)
            if success:
                item = (curr_context, encoded_image.tobytes(), time())
                

                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # Remove the oldest item
                    except queue.Empty:
                        pass  # In case it got empty suddenly
                frame_queue.put_nowait(item)

        # Show live video
        cv2.imshow("Live Camera", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("👋 Quit camera thread.")
            exit()
        elif key & 0xFF == ord('r'):
            recognize_person_thread = threading.Thread(target=recognize_faces, daemon=True)
            recognize_person_thread.start()
        elif key & 0xFF == ord('s'):
            add_person_thread = threading.Thread(target=add_person, daemon=True)
            add_person_thread.start()
        elif key & 0xFF == ord('d'):
            del_person_thread = threading.Thread(target=delete_user, daemon=True)
            del_person_thread.start()
            
            

    cap.release()
    cv2.destroyAllWindows()



def get_cropped_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    return faces



def get_input(prompt):
    name = "xyztpr2"
    while not name or name == "xyztpr2":
        name = input(prompt)
        
    return name





def post_request(url, data: dict, files: list):
    global last_time, rate_limit

    elapsed = time() - last_time
    if elapsed < rate_limit and last_time != 0:
        return {}

    last_time = time()

    # Instead of dict, prepare a list of tuples
    files_payload = []

    for key, (filename, file_bytes, content_type) in files:
        file_stream = io.BytesIO(file_bytes)
        files_payload.append((key, (filename, file_stream, content_type)))

    response = requests.post(url, data=data, files=files_payload, allow_redirects=True)
    # print(response.text)
    return response.json()

    
    


def capture_faces(context, start_time,max_faces=5,rec_mode=False):
    images = []

    while len(images) < max_faces:
        matched = []

        # Pull items but don't exceed max_faces
        while not frame_queue.empty() and len(images) + len(matched) < max_faces:
            ctx, img_bytes, timestamp = frame_queue.get()
            if timestamp<start_time: break
            # if ctx == context:
            matched.append(('images', ('face.jpg', img_bytes, 'image/jpeg')))

        images.extend(matched)

        if len(images) >= max_faces:
            break

        
        sleep(0.3)  

    return images





def add_person():
    if activity_lock.locked(): return
    with activity_lock:
        
        global curr_context
        
        try:
            print("Capturing images for new user ")
            
            name = get_input("Please Enter the name of the person- ")
            change_context("add_person")
            print("New user please stand facing the camera")
            face_images = capture_faces("add_person", time(), 120)  # Show window only during this action
            print(f"Captured {len(face_images)} faces for {name}")
            response = post_request(
                add_face_url,
                files=face_images,
                data={"name": name, "type": "upload"}
            )

            if response.get("status") == "success":
                print(f" added user {response.get('name','')} with number of images {len(response.get('saved_faces',[]))}")
            else:
                print(f"Failed to upload faces. Response: {response}")
        except Exception as e:
            print("Error in add person",e)
        change_context("default")

def delete_user(timeout=45):
    if activity_lock.locked(): return
    with activity_lock:
        global curr_context
        
        name = get_input("Please Enter the name of person to delete- ")
        start_time = time()
        
        print("Needs Authentication before Deleting. Please be in front of camera")
        try:
            while True:
                
                change_context("delete_user")
                
                rec = recognize_faces("delete_user")
                if rec:
                    res = post_request(base_url, files=[],data={"type": "delete", "name": name})
                    print(f"Deleted user {name}")
                    
                    break
                if time() - start_time > timeout:
                    print("Could not authenticate user. Timeout occurred")
                    break
        except Exception as e:
            print(f"Error in delete_user {e}")
        
        change_context("deafault")

def recognize_faces(context = "default"):
    
    print("Attempting to authenticate")
        
    try:
        face_arr= capture_faces(context,time(),5)
        # print("face_arr", face_arr)
        result = post_request(recognize_face_url, files=face_arr, data={"type": "recognize"})
        if result:
            print(result)
        result = result.get("recognized_faces", [{}, {}])
        score = result[0].get("max_score", 0.0)
        name = result[0].get("name", "None")
        if score > 0.50:
            print(f"Member {name} recognized")
            return True
        
    except Exception as e:
        print(e)
    
    print("Authentication Failed ")
    return False


    def input_thread():
        while True:
            inp = get_input("1 for adding a person and 2 for deleting a user ")
            
            if inp=="1": add_person()
            if inp=="2": delete_user()


if __name__ == "__main__":
    with open("server_url.txt","r") as f:
        base_url = f.readline()
    print(base_url)
    recognize_face_url= base_url
    add_face_url = base_url
    camera_thread_func()
    # Start camera thread
    # cam_thread = threading.Thread(target=camera_thread_func, daemon=True)
    # cam_thread.start()
    # input_thread = threading.Thread(target=input_thread, daemon=True)
    # input_thread.start()

    # input_thread.join()
    # cam_thread.join()
    sleep(2) 
    

