import cv2
import numpy as np
import time
import datetime


def main():
    # Adjustable parameters
    CAMERA_INDEX = 0  # 0 is usually the default camera
    CHANGE_THRESHOLD = 25  # Threshold for pixel difference to be considered a change
    MIN_CHANGED_PERCENTAGE = 7  # Minimum percentage of changed pixels to trigger a lap
    COOLDOWN_SECONDS = 3  # Cooldown period after detecting a lap

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Wait a bit for the camera to initialize
    time.sleep(1)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        return

    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    # Initialize variables for lap timing
    lap_start_time = time.perf_counter()
    lap_count = 0
    lap_times = []
    cooldown = False
    cooldown_end_time = 0

    # Set up file for saving lap times
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    lap_times_filename = f"lap_times_{timestamp}.txt"

    with open(lap_times_filename, 'w') as f:
        f.write("Lap Times\n")
        f.write("=========\n")
        f.write(f"Session started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    print(f"Lap times will be saved to: {lap_times_filename}")
    print("Lap timer started. Press 'q' to quit.")

    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        current_time = time.perf_counter()

        # Check if cooldown period is over
        if cooldown and current_time >= cooldown_end_time:
            cooldown = False
            print("Cooldown period ended, ready for next lap.")

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Calculate absolute difference from previous frame
        frame_delta = cv2.absdiff(prev_gray, gray)

        # Threshold the difference
        thresh = cv2.threshold(frame_delta, CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Calculate percentage of changed pixels
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        changed_percentage = (changed_pixels / total_pixels) * 100

        # Display information on frame
        cv2.putText(frame, f"Lap: {lap_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_lap_time = current_time - lap_start_time
        cv2.putText(frame, f"Time: {current_lap_time:.2f}s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Change: {changed_percentage:.2f}%", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cooldown:
            cv2.putText(frame, "COOLDOWN", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame and difference
        cv2.imshow('Lap Timer', frame)
        cv2.imshow('Difference', thresh)

        # Detect significant change (car passing)
        if changed_percentage > MIN_CHANGED_PERCENTAGE and not cooldown:
            lap_end_time = current_time
            lap_duration = lap_end_time - lap_start_time

            lap_count += 1
            lap_times.append(lap_duration)

            print(f"Lap {lap_count} completed in {lap_duration:.2f} seconds")

            # Save lap time to file
            with open(lap_times_filename, 'a') as f:
                f.write(f"Lap {lap_count}: {lap_duration:.2f} seconds\n")

            # Start new lap timing
            lap_start_time = lap_end_time

            # Set cooldown
            cooldown = True
            cooldown_end_time = current_time + COOLDOWN_SECONDS
            print(f"Cooldown period started ({COOLDOWN_SECONDS} seconds).")

        # Update previous frame
        prev_gray = gray

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save final statistics to file
    with open(lap_times_filename, 'a') as f:
        f.write("\nSession Summary\n")
        f.write("==============\n")
        f.write(f"Total laps: {lap_count}\n")
        if lap_times:
            f.write(f"Fastest lap: {min(lap_times):.2f} seconds (Lap {lap_times.index(min(lap_times)) + 1})\n")
            f.write(f"Average lap time: {sum(lap_times) / len(lap_times):.2f} seconds\n")
        f.write(f"Session ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Print all lap times when done
    print("\nAll lap times:")
    for i, lap_time in enumerate(lap_times, 1):
        print(f"Lap {i}: {lap_time:.2f} seconds")

    if lap_times:
        print(f"\nFastest lap: {min(lap_times):.2f} seconds (Lap {lap_times.index(min(lap_times)) + 1})")
        print(f"Average lap time: {sum(lap_times) / len(lap_times):.2f} seconds")

    print(f"\nLap times saved to: {lap_times_filename}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()