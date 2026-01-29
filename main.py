import cv2
from camera import ThreadedCamera
from surveillance import SurveillanceSystem

def main():
    # 5. Performance Optimization: Start Threaded Camera
    camera = ThreadedCamera()
    camera_stream = camera.start()
    
    if camera_stream is None:
        print("Failed to start camera.")
        return

    system = SurveillanceSystem()
    
    print("Industrial Monitoring System Started...")
    print("Press 'ESC' or 'q' to exit.")

    while True:
        try:
            frame = camera_stream.get_frame()
            if frame is None:
                continue

            # Process frame (Detection)
            # We ignore the second return value (mask) as we are using YOLO now
            processed_frame, _ = system.process_frame(frame)
            
            # Show feeds
            # Show feeds
            if processed_frame is not None:
                window_name = "Industrial Monitoring (YOLOv8)"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, processed_frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
                
        except AttributeError:
            pass
        except KeyboardInterrupt:
            break

    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
