import cv2
import numpy as np

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
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, adjust if necessary
    tracker = ObjectTracker()

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", mouse_callback, (tracker, None))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if tracker.tracking:
            for obj in tracker.objects:
                success, bbox = obj['tracker'].update(frame)
                if success:
                    obj['bbox'] = tuple(map(int, bbox))
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame, obj['label'], (p1[0], p1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            tracker.tracking = False
        elif key == ord('d'):
            tracker.tracking = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()