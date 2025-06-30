from flask import Blueprint, render_template, request, session, redirect, url_for, Response, current_app
from flask_login import login_required, current_user
import threading
from tensorflow.keras.models import load_model
from .person_tracker import PersonTracker
import os
import numpy as np
import cv2
from .models import Detection
from . import db
from datetime import datetime

views = Blueprint('views', __name__)

@views.route('/')
def index():
    return render_template("index.html",user=current_user)

@views.route('/home')
@login_required
def home():
    return render_template("home.html",user=current_user)


#### Real Time Detection Views and functions ####

@views.route('/monitor', methods=['GET', 'POST'])
def monitor():
    if request.method == 'POST':
        source = request.form['source']
        camera_angle = request.form['camera_angle']
        username = current_user.name
        phone_number = current_user.alert_phone_number

        with thread_lock:
            monitoring_event.set()
            print("Monitoring SET")

        thread = threading.Thread(
            target=monitoring_loop,
            args=(source,username, phone_number, camera_angle,current_user.id)
        )
        thread.start()

        start_periodic_commit(current_app._get_current_object())

        return render_template('test.html', user=current_user)
    return render_template('test.html', user=current_user)

@views.route('/video_feed')
@login_required
def video_feed():
    print("Generating Stream")
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route('/stop_monitoring')
def stop_monitoring():
    with thread_lock:
        monitoring_event.clear()
    return redirect(url_for('views.monitor'))

def generate_stream():
    global frame_to_display

    while True:
        with thread_lock:
            if not monitoring_event.is_set():
                print("Monitoring OFF. Stopping stream.")
                break

        if frame_to_display is not None:
            ret, buffer = cv2.imencode('.jpg', frame_to_display)
            if ret:
                
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                print("Frame encoding failed.")
    



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


cctv_model_path = os.path.join(BASE_DIR, "models", "pose_violence_detection_cctv.keras")
front_model_path = os.path.join(BASE_DIR, "models", "pose_transformer.keras")

deepsort_ckpt = os.path.join(BASE_DIR, 'deep_sort', 'deep', 'checkpoint', 'ckpt.t7')
pose_model_cctv = load_model(cctv_model_path)
pose_model_front = load_model(front_model_path)


monitoring_event = threading.Event()
frame_to_display = None
thread_lock = threading.Lock()
save_dir = "static/screenshots"
os.makedirs(save_dir, exist_ok=True)

detection_buffer = []

def monitoring_loop(source,username,alert_phone, cam_view,user_id):
    global frame_to_display, detection_buffer
    
    print("Monitoring Loop Started")
    pose_model = pose_model_front if cam_view == "0" else pose_model_cctv
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    tracker = PersonTracker(deepsort_ckpt=deepsort_ckpt, buffer_size=20)

    violence_counter = 0
    already_logged = False

    print("Monitoring event set: ",monitoring_event.is_set())
    print("Camera Opened: ",cap.isOpened())

    label_detection = ""
    labels = []

    while monitoring_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame, label= tracker.update(frame, violence_model=pose_model)
        
        labels.append(label)
        
        if len(labels) == 20:
            label_detection = labels[19]
            print(labels)
            print(label_detection)
            labels = []

        if label_detection == "Violence":
            violence_counter += 1
            label_detection = ""

        elif label_detection == "NonViolence":
            violence_counter = 0
            already_logged = False
        
        if violence_counter >= 3 and not already_logged:

            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection_data = f"Violence detected at {timestamp}"
            filename = f"user{user_id}_{timestamp}.jpg"
            full_path = os.path.join(save_dir, filename)

            success = cv2.imwrite(full_path, frame)
            print(f"Violence Counter: {violence_counter}, already logged: {already_logged}")

            if success:
                detection_buffer.append({
                'data': detection_data,
                'user_id': user_id,
                'timestamp': datetime.now(),
                "image_path": full_path
                })

            violence_counter = 0
            already_logged = True
            label_detection = ""
        

        frame_to_display = frame

    cap.release()

def commit_detections():
    global detection_buffer

    if detection_buffer:
        with current_app.app_context(): 
            for detection in detection_buffer:
                new_detection = Detection(
                    data=detection['data'],
                    user_id=detection['user_id'],
                    date=detection['timestamp'],
                    image_path=detection['image_path']
                )
                db.session.add(new_detection)
            db.session.commit() 
            print(f"Committed {len(detection_buffer)} detections to database.")
        
        detection_buffer = []

def start_periodic_commit(app, interval=20):
    def commit_detections():
        with app.app_context():  
            global detection_buffer
            while detection_buffer:
                detection = detection_buffer.pop(0)
                new_detection = Detection(
                    data=detection['data'],
                    user_id=detection['user_id'],
                    date=detection['timestamp'],
                    image_path=detection.get('image_path')
                )
                db.session.add(new_detection)
            db.session.commit()

        threading.Timer(interval, start_periodic_commit, args=(app, interval)).start()

    threading.Thread(target=commit_detections).start()

