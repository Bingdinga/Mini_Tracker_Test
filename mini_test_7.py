import cv2
import numpy as np
from picamera2 import Picamera2
import time
import random
import sys
import threading

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
    hash_value = hash(label)
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return (r, g, b)

def detect_objects(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 100  # Adjust this value to filter out small contours
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    return filtered_contours

def draw_detected_objects(frame, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def print_menu():
    print("\nDnD Mini Tracker Menu:")
    print("1. Start tracking")
    print("2. Stop tracking")
    print("3. Zoom in")
    print("4. Zoom out")
    print("5. Toggle autofocus")
    print("6. Quit")
    print("Enter your choice: ", end="", flush=True)

def handle_input(tracker, zoom_factor, running, picam2):
    autofocus_enabled = True
    while running[0]:
        print_menu()
        choice = input().strip()
        if choice == '1':
            tracker.tracking = True
            print("Tracking started")
        elif choice == '2':
            tracker.tracking = False
            print("Tracking stopped")
        elif choice == '3':
            zoom_factor[0] = min(4.0, zoom_factor[0] + 0.1)
            print(f"Zoomed in. Zoom factor: {zoom_factor[0]:.1f}")
        elif choice == '4':
            zoom_factor[0] = max(0.1, zoom_factor[0] - 0.1)
            print(f"Zoomed out. Zoom factor: {zoom_factor[0]:.1f}")
        elif choice == '5':
            autofocus_enabled = not autofocus_enabled
            if autofocus_enabled:
                picam2.set_controls({"AfMode": 2})  # Continuous autofocus
                print("Autofocus enabled")
            else:
                picam2.set_controls({"AfMode": 0})  # Manual focus
                print("Autofocus disabled")
        elif choice == '6':
            running[0] = False
            print("Quitting...")
        else:
            print("Invalid choice. Please try again.")

def main():
    picam2 = None
    try:
        picam2 = Picamera2()
        
        config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)})
        picam2.configure(config)
        
        picam2.set_controls({"AfMode": 2})  # Enable continuous autofocus
        
        picam2.start()
        
        time.sleep(2)  # Wait for the camera to warm up

        tracker = ObjectTracker()

        cv2.namedWindow("Tracking")

        zoom_factor = [1.0]
        running = [True]

        input_thread = threading.Thread(target=handle_input, args=(tracker, zoom_factor, running, picam2))
        input_thread.daemon = True
        input_thread.start()

        while running[0]:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            if zoom_factor[0] != 1.0:
                h, w = frame.shape[:2]
                zoom_h, zoom_w = int(h / zoom_factor[0]), int(w / zoom_factor[0])
                start_y, start_x = (h - zoom_h) // 2, (w - zoom_w) // 2
                frame = frame[start_y:start_y+zoom_h, start_x:start_x+zoom_w]
                frame = cv2.resize(frame, (w, h))

            # Detect and draw potential objects
            contours = detect_objects(frame)
            draw_detected_objects(frame, contours)

            # Resize frame for display
            display_frame = cv2.resize(frame, (960, 540))
            cv2.imshow("Tracking", display_frame)
            cv2.waitKey(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if picam2:
            picam2.stop()
        cv2.destroyAllWindows()
        print("Script terminated. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()