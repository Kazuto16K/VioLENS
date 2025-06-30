import cv2
import numpy as np
import pandas as pd
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp
import logging

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

class PersonTracker:
    def __init__(self, yolo_weights="yolov8n.pt", deepsort_ckpt="deep_sort/deep/checkpoint/ckpt.t7", buffer_size=20):
        self.current_video_predictions = []
        self.yolo_model = YOLO(yolo_weights)#.to('cuda')
        self.tracker = DeepSort(model_path=deepsort_ckpt, max_age=70) #,use_cuda=True
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pose = mp.solutions.pose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
        self.person_buffers = defaultdict(lambda: deque(maxlen=buffer_size))
        self.predictions = {}  # track_id: prediction
        self.buffer_size = buffer_size

    def make_landmark_timestep(self, results):
        return [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)]
    
    def normalize_pose_with_visibility(self,pose_vector):
        pose = pose_vector.reshape((33, 4))
        xyz = pose[:, :3]  
        visibility = pose[:, 3:]  

        # Use shoulders and hips for torso center & scale
        left_shoulder, right_shoulder = xyz[11], xyz[12]
        left_hip, right_hip = xyz[23], xyz[24]

        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        center = (mid_shoulder + mid_hip) / 2

        torso_length = np.linalg.norm(mid_shoulder - mid_hip)
        if torso_length == 0:
            torso_length = 1e-6

        normalized_xyz = (xyz - center) / torso_length
        normalized_pose = np.concatenate([normalized_xyz, visibility], axis=1)  
        return normalized_pose.flatten()
    
    def predict_violence(self, sequence, track_id, model):
        try:
            pred = model.predict(sequence, verbose=0)
            label = "NonViolence" if pred[0][0] > 0.5 else "Violence"
            self.current_video_predictions.append(label)
            self.predictions[track_id] = label
            self.person_buffers[track_id].clear()
        except Exception as e:
            print(f"[Error] Prediction failed for ID {track_id}: {e}")

    def update(self, frame, violence_model=None):
        og_frame = frame.copy()
        detections = self.yolo_model(frame, classes=0, conf=0.65,verbose=False)[0]
        boxes = detections.boxes

        if boxes is not None and len(boxes) > 0:
            xywh = boxes.xywh.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            self.tracker.update(xywh, conf, frame)
            #self.tracker.update(xywh, frame)

            for track in self.tracker.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_tlbr())

                # Clip coordinates to frame bounds
                #margin = 20
                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))

                # Skip if bounding box is invalid
                if x2 <= x1 or y2 <= y1:
                    continue
                
                person_crop = frame[y1:y2, x1:x2]

                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                results = self.pose.process(person_rgb)

                if results.pose_landmarks:
                    # Real-time keypoints drawing

                    landmarks = results.pose_landmarks.landmark
                    for connection in mp.solutions.pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(landmarks) and end_idx < len(landmarks):
                            x1_conn = int(landmarks[start_idx].x * (x2 - x1)) + x1
                            y1_conn = int(landmarks[start_idx].y * (y2 - y1)) + y1
                            x2_conn = int(landmarks[end_idx].x * (x2 - x1)) + x1
                            y2_conn = int(landmarks[end_idx].y * (y2 - y1)) + y1
                            cv2.line(og_frame, (x1_conn, y1_conn), (x2_conn, y2_conn), (255, 255, 255), 1)

                    for lm in landmarks:
                        cx = int(lm.x * (x2 - x1)) + x1
                        cy = int(lm.y * (y2 - y1)) + y1
                        cv2.circle(og_frame, (cx, cy), 3, (0, 255, 0), -1)
                    
                    lm = self.make_landmark_timestep(results)
                    normalized_lm = self.normalize_pose_with_visibility(np.array(lm, dtype=np.float32))
                    self.person_buffers[track_id].append(normalized_lm)

                    # Violence prediction after buffer is full
                    if len(self.person_buffers[track_id]) == self.buffer_size and violence_model:
                        sequence = np.array(self.person_buffers[track_id], dtype=np.float32)[np.newaxis, ...]
                        print(f"Sequence shape: {sequence.shape}, dtype: {sequence.dtype}")

                        pred = violence_model.predict(sequence, verbose=0)
                        print(pred)
                        label = "NonViolence" if pred[0][0] > 0.5 else "Violence"
                        self.current_video_predictions.append(label)
                        self.predictions[track_id] = label
                        self.person_buffers[track_id].clear()
                        #self.executor.submit(self.predict_violence, sequence, track_id, violence_model)

                # Draw bounding box
                color = (255, 0, 0)
                cv2.rectangle(og_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(og_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw label
                label = self.predictions.get(track_id, "P R O C E S S I N G")
                label_color = (255, 0, 255) if label == "P R O C E S S I N G" else (0, 255, 0) if label == "NonViolence" else (0, 0, 255)
                cv2.putText(og_frame, label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        return og_frame
