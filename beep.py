import cv2
import numpy as np
import time
import platform

# Platform-specific audio imports
if platform.system() == "Windows":
    import winsound
else:
    import os
    import subprocess

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read initial frames
ret, prev_frame = cap.read()
ret, curr_frame = cap.read()

# Motion detection parameters
MOTION_THRESHOLD = 5000  # Sensitivity adjustment - decrease for more sensitivity
BEEP_DURATION = 200  # milliseconds
BEEP_FREQUENCY = 1000  # Hz

# Cooldown period to prevent repeated alerts
last_alert_time = 0
alert_cooldown = 1  # seconds

print("Motion detection system activated. Press 'q' to exit.")

while cap.isOpened():
    # Convert frames to grayscale
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and false positives
    gray_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
    gray_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)
    
    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(gray_prev, gray_curr)
    
    # Apply threshold to highlight significant changes
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Dilate threshold image to fill holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Count non-zero pixels (measure of motion intensity)
    motion_level = np.count_nonzero(thresh)
    
    # Display video feed with motion indicator
    display_frame = curr_frame.copy()
    cv2.putText(display_frame, f"Motion Level: {motion_level}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Check if motion exceeds threshold
    current_time = time.time()
    if motion_level > MOTION_THRESHOLD:
        cv2.putText(display_frame, "MOTION DETECTED!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Trigger alert with cooldown period
        if current_time - last_alert_time > alert_cooldown:
            if platform.system() == "Windows":
                winsound.Beep(BEEP_FREQUENCY, BEEP_DURATION)
            elif platform.system() == "Darwin":  # macOS
                os.system('afplay /System/Library/Sounds/Ping.aiff &')
            elif platform.system() == "Linux":
                os.system('aplay /usr/share/sounds/speech-dispatcher/test.wav &')
            
            last_alert_time = current_time
    
    # Display camera feed
    cv2.imshow("Motion Detector", display_frame)
    cv2.imshow("Threshold View", thresh)  # Optional: show threshold image
    
    # Update frames
    prev_frame = curr_frame
    ret, curr_frame = cap.read()
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()