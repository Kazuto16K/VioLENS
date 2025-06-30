import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from person_tracker2 import PersonTracker

# Load your violence detection model
model = load_model("pose_violence_detection_transformer_2.keras", custom_objects={'Orthogonal': tf.keras.initializers.Orthogonal})
print("Model loaded. Input shape:", model.input_shape)

# Initialize video
cap = cv2.VideoCapture(0)  # or 1 if external webcam

# Initialize tracker
tracker = PersonTracker(buffer_size=20)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = tracker.update(frame, violence_model=model)

    cv2.imshow("Multi-Person Violence Detection", output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
