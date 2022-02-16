import cv2
 
frames_Array = []

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),

    15.,
    (640,480))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    

    #       frames_Array.append(frame) 
    #       print(frames_Array)

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))

    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)

    print(out.write(frame.astype('uint8')))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)