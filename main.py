import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Read class names from coco.names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Paths to the configuration and weights files
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

confThreshold = 0.5

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform object detection
    classIds, confs, bbox = net.detect(img, confThreshold=confThreshold)
    print(classIds, bbox)
    
    # Draw bounding boxes and class labels on the image
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId - 1 < len(classNames):
                label = f'{classNames[classId - 1]}: {confidence:.2f}'
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, label.upper(), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                print(f"Class ID {classId} is out of range")

    # Display the output image
    cv2.imshow("Output", img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()