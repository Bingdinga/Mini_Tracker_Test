import cv2
import numpy as np
from picamera2 import Picamera2
import time
import libcamera
import random

class ObjectTracker:
    def __init__(self):
        self.objects = []
        self.tracking = False
        self.current_id = 0

    def add_object(self, x, y, label):
        self.objects.append({
            'id': self.current_id,
            'label': label,
            'bbox': None,
            'tracker': cv2.TrackerKCF_create()
        })
        self.current_id += 1

    def update_label(self, object_id, new_label):
        for obj in self.objects:
            if obj['id'] == object_id:
                obj['label'] = new_label
                break

    def remove_object(self, object_id):
        self.objects = [obj for obj in self.objects if obj['id'] != object_id]

def get_object_color(label):
    # Generate a pseudo-random color based on the label
    hash_value = hash(label)
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return (r, g, b)

def create_particle_effect(frame, bbox, color):
    x, y, w, h = bbox
    for _ in range(20):  # Number of particles
        start_x = random.randint(x, x + w)
        start_y = random.randint(y, y + h)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(5, 15)
        end_x = int(start_x + length * np.cos(angle))
        end_y = int(start_y + length * np.sin(angle))
        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 1)

def draw_pretty_object(frame, label, bbox, color):
    x, y, w, h = bbox

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Add particle effect
    create_particle_effect(frame, bbox, color)

    # Add label
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def mouse_callback(event, x, y, flags, param):
    tracker, frame = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if tracker.tracking:
            # Check if click is on an existing object
            for obj in tracker.objects:
                if obj['bbox'] and obj['bbox'][0] < x < obj['bbox'][0] + obj['bbox'][2] and obj['bbox'][1] < y < obj['bbox'][1] + obj['bbox'][3]:
                    new_label = input(f"Enter new label for object {obj['id']} (current: {obj['label']}): ")
                    tracker.update_label(obj['id'], new_label)
                    return
        # If not on existing object, add new object
        label = input("Enter label for new object: ")
        tracker.add_object(x, y, label)
        bbox = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
        tracker.objects[-1]['tracker'].init(frame, bbox)
        tracker.objects[-1]['bbox'] = bbox

def main():
    try:
        # Initialize PiCamera2
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
        picam2.start()
        
        time.sleep(2)  # Wait for the camera to warm up

        tracker = ObjectTracker()

        cv2.namedWindow("Tracking")
        cv2.setMouseCallback("Tracking", mouse_callback, (tracker, None))

        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            if tracker.tracking:
                for obj in tracker.objects:
                    success, bbox = obj['tracker'].update(frame)
                    if success:
                        obj['bbox'] = tuple(map(int, bbox))
                        color = get_object_color(obj['label'])
                        draw_pretty_object(frame, obj['label'], obj['bbox'], color)

            cv2.imshow("Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                tracker.tracking = False
            elif key == ord('d'):
                tracker.tracking = True

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()

if __name__ == "__main__":
    main()