import sys
import cv2 
import imutils
import numpy as np
from yoloDet import YoloTRT


model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

cap1 = cv2.VideoCapture("videos/car.mp4")
cap2 = cv2.VideoCapture("videos/car.mp4")
#cap1 = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture("rtsp://admin:Install01@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0")

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(f'x: {x}, y: {y}')
        
cv2.namedWindow('Combined Output')
cv2.setMouseCallback('Combined Output', POINTS)

area_1 = [(50, 257), (299, 257), (300, 290), (5, 290)]
area_2 = [(350, 200), (510, 200), (554, 218), (365, 218)]

bbox_thickness = 2

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 and not ret2:
        break

    # Resize frames for consistent display size
    if ret1:
        frame1 = imutils.resize(frame1, width=640)
    if ret2:
        frame2 = imutils.resize(frame2, width=640)

    # Perform inference on frame 1
    if ret1:
        detections1, t = model.Inference(frame1)
        
        cv2.putText(frame1, "Frame1", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Print detections to check the structure
        print("Detections (video 1):", detections1)

        # Draw the polygon
        cv2.polylines(frame1, [np.array(area_1, np.int32)], True, (0, 255, 0), 2)

        # Iterate over detections and draw center point
        for det in detections1:
            box = det.get('box')
            if box is not None:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box with specified thickness
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (255, 0, 0), bbox_thickness)

                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Print center point for debugging
                print(f"Center (video 1): ({center_x}, {center_y})")

                # Draw center point
                cv2.circle(frame1, (center_x, center_y), 3, (0, 0, 255), -1)

                # Check if center point is inside the area
                if cv2.pointPolygonTest(np.array(area_1, np.int32), (center_x, center_y), False) >= 0:
                    cv2.putText(frame1, "car in area", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Perform inference on frame 2
    if ret2:
        detections2, t = model.Inference(frame2)
    
        cv2.putText(frame2, "Frame2", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Print detections to check the structure
        print("Detections (video 2):", detections2)

        # Draw the polygon
        cv2.polylines(frame2, [np.array(area_2, np.int32)], True, (0, 255, 0), 2)

        # Iterate over detections and draw center point
        for det in detections2:
            box = det.get('box')
            if box is not None:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box with specified thickness
                cv2.rectangle(frame2, (x1, y1), (x2, y2), (255, 0, 0), bbox_thickness)

                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Print center point for debugging
                print(f"Center (video 2): ({center_x}, {center_y})")

                # Draw center point
                cv2.circle(frame2, (center_x, center_y), 3, (0, 0, 255), -1)

                # Check if center point is inside the area
                if cv2.pointPolygonTest(np.array(area_2, np.int32), (center_x, center_y), False) >= 0:
                    cv2.putText(frame2, "car in area", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Ensure both frames have the same dimensions and type before concatenation
    if ret1 and ret2:
        frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))  # Resize frame 1 to match frame 2

        # Combine frames horizontally using cv2.hconcat
        combined_frame = cv2.hconcat([frame1, frame2])

        # Display combined frame
        cv2.imshow("Combined Output", combined_frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
