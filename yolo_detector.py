from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2 as cv

model = YOLO("yolov8n.pt")

def display_text(frame):
    text = "Jaturawich-Clicknext-Internship-2024"
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 255)  
    
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_x = frame.shape[1] - text_size[0] - 10 
    text_y = 30  

    cv.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)



def draw_boxes(frame, boxes):
    """Draw detected bounding boxes and tracking line on image frame"""
    cat_class_id = 15  
    cat_positions = []

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        if int(class_id) != cat_class_id:
            continue  # Skip if the others class object
        
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        blue_color = (255,0,0)

        # Draw bounding box
        annotator.box_label(
            box=coordinator, label=class_name, color=blue_color
        )


    display_text(frame)

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.predict(frame)

    for result in results:
        if result.boxes:  # Ensure there are boxes to process
            frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect objects from image frame
            frame_result = detect_object(frame)

            # Show result
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", frame_result)
            cv.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv.destroyAllWindows()
