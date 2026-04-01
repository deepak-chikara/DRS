from pitch import pitch
from batsman import batsman_detect
from ball_detect import ball_detect
import numpy as np
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder

# ===== GLOBAL VARIABLES =====
pitch_point = None
impact_point = None
impact_locked = False
pitch_counter = 0

trajectory_points = []

mycolorFinder = ColorFinder(False)
hsvVals = {
    "hmin": 10,
    "smin": 44,
    "vmin": 192,
    "hmax": 125,
    "smax": 114,
    "vmax": 255,
}

tuned_rgb_lower = np.array([112, 0, 181])
tuned_rgb_upper = np.array([255, 255, 255])

tuned_canny_threshold1 = 100
tuned_canny_threshold2 = 200

cap = cv2.VideoCapture(r"lbw.mp4")  # Path to your video file
# cap = cv2.VideoCapture("http://your-ip:port/video")
#cap = cv2.VideoCapture(0)   # for webcam
cap2 = cv2.VideoCapture(1)   # side camera

# ===== VIDEO CONTROL =====
paused = False
frame_pos = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# =========================

def ball_pitch_pad(x, x_prev, prev_x_diff, y, y_prev, prev_y_diff, batLeg):
    if x_prev == 0 and y_prev == 0:
        return "Motion", 0, 0

    if abs(x - x_prev) > 3 * abs(prev_x_diff) and abs(prev_x_diff) > 0:
        if y < batLeg:
            return "Pad", x - x_prev, y - y_prev

    if y - y_prev < 0 and prev_y_diff > 0:
        if y < batLeg:
            return "Pad", x - x_prev, y - y_prev
        else:
            return "Pitch", x - x_prev, y - y_prev

    return "Motion", x - x_prev, y - y_prev


x = y = 0
batLeg = 0
x_prev = y_prev = 0
prev_x_diff = prev_y_diff = 0
lbw_detected = False

while True:
    x_prev = x
    y_prev = y

    # ===== SMOOTH BALL =====
    if x != 0 and y != 0 and x_prev != 0 and y_prev != 0:
        x = int(0.6 * x_prev + 0.4 * x)
        y = int(0.6 * y_prev + 0.4 * y)
    # =======================

    # ===== REMOVE NOISE =====
    if x_prev != 0 and abs(x - x_prev) > 50:
        x = x_prev
    if y_prev != 0 and abs(y - y_prev) > 50:
        y = y_prev
    # ========================

    # ===== VIDEO CONTROL =====
    if not paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        frame, img = cap.read()
        frame_pos += 1
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        frame, img = cap.read()

    if not frame:
        break

    pitchImg = img.copy()
    batsmanImg = img.copy()

    # ===== BALL DETECTION =====
    ballContour, x, y = ball_detect(img, mycolorFinder, hsvVals)


    # ===== CLEAN TRAJECTORY =====
    if x != 0 and y != 0:
        trajectory_points.append((x, y))
    else:
        trajectory_points.clear()   # reset if ball lost
    
    if len(trajectory_points) > 10:
        trajectory_points.pop(0)
    # ============================

    all = ballContour.copy() if ballContour is not None else img.copy()

    # ===== BATSMAN =====
    batsmanContours = batsman_detect(
        img,
        tuned_rgb_lower,
        tuned_rgb_upper,
        tuned_canny_threshold1,
        tuned_canny_threshold2,
    )

    # ===== PITCH =====
    pitchContour = pitch(img)

    # ===== FINAL STABLE PITCH DETECTION =====
    if pitch_point is None and x != 0 and y != 0:

        for cnt in pitchContour:
            if cv2.contourArea(cnt) > 50000:

                inside = cv2.pointPolygonTest(cnt, (x, y), False)

                if inside >= 0:
                    pitch_counter += 1
                else:
                    pitch_counter = 0

                # detect only after stable presence
                if pitch_counter > 5:
                    pitch_point = (x, y)
                    print("Pitch detected!")
                    break
    # ==========================================

    # ===== DRAW PITCH =====
    for cnt in pitchContour:
        if cv2.contourArea(cnt) > 50000:
            cv2.drawContours(pitchImg, cnt, -1, (0, 255, 0), 10)
            cv2.drawContours(all, cnt, -1, (0, 255, 0), 10)

    # ===== STUMPS =====
    if len(pitchContour) > 0:
        pitch_cnt = max(pitchContour, key=cv2.contourArea)
        px, py, w_box, h_box = cv2.boundingRect(pitch_cnt)

        center_x = px + w_box // 2
        stump_x1 = center_x - int(w_box * 0.05)
        stump_x2 = center_x + int(w_box * 0.05)

        cv2.line(all, (stump_x1, 0), (stump_x1, img.shape[0]), (255, 255, 0), 2)
        cv2.line(all, (stump_x2, 0), (stump_x2, img.shape[0]), (255, 255, 0), 2)

    # ===== BATSMAN DRAW =====
    current_batLeg = 10000

    for cnt in batsmanContours:
        if cv2.contourArea(cnt) > 5000:
            if (min(cnt[:, :, 1]) < y and y != 0):
                batLeg_candidate = max(cnt[:, :, 1])
                if batLeg_candidate < current_batLeg:
                    current_batLeg = batLeg_candidate

                cv2.drawContours(batsmanImg, cnt, -1, (0, 0, 255), 10)

    batLeg = current_batLeg

    # ===== IMPACT DETECTION =====
    if not impact_locked and x != 0 and y != 0 and batLeg != 10000:
        for cnt in batsmanContours:
            if cv2.contourArea(cnt) > 5000:
                dist = cv2.pointPolygonTest(cnt, (x, y), True)
                if dist >= -15 and abs(y - batLeg) < 25:
                    impact_point = (x, y)
                    impact_locked = True
                    lbw_detected = True
                    print("Impact detected!")
                    break

    # ===== DRAW POINTS =====
    if pitch_point is not None:
        cv2.circle(all, pitch_point, 8, (255, 0, 0), -1)
        cv2.putText(all, "Pitch", pitch_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # ===== DRAW TRAJECTORY =====
    for i in range(1, len(trajectory_points)):
        cv2.line(all, trajectory_points[i-1], trajectory_points[i], (0,255,255), 2)
    # =========================== 

    # ===== CORRECT TRAJECTORY PREDICTION =====
    if len(trajectory_points) >= 5 and impact_point is not None:
    
        y_vals = [p[1] for p in trajectory_points]
        x_vals = [p[0] for p in trajectory_points]
    
        try:
            # Fit curve: y → x (correct direction)
            curve = np.polyfit(y_vals, x_vals, 2)
    
            # Predict future downward motion
            future_y = np.linspace(y_vals[-1], y_vals[-1] + 200, 20)
            future_x = np.polyval(curve, future_y)
    
            # Draw predicted path
            for i in range(len(future_x)-1):
                pt1 = (int(future_x[i]), int(future_y[i]))
                pt2 = (int(future_x[i+1]), int(future_y[i+1]))
                cv2.line(all, pt1, pt2, (255, 0, 255), 2)
    
            # ===== STUMP CHECK =====
            hit_stumps = False
    
            for fx, fy in zip(future_x, future_y):
                if stump_x1 <= fx <= stump_x2:
                    hit_stumps = True
                    break
    
            if hit_stumps:
                cv2.putText(all, "HITTING STUMPS", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                cv2.putText(all, "MISSING STUMPS", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    
        except:
            pass
    # ============================================

    # =================================

    if impact_point is not None:
        cv2.circle(all, impact_point, 10, (0, 0, 255), -1)
        cv2.putText(all, "Impact", impact_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 

    # ===== DISPLAY =====
    imgStack = cvzone.stackImages([
        ballContour if ballContour is not None else img,
        pitchImg,
        batsmanImg,
        all
    ], 2, 0.5)

    cv2.putText(imgStack, "SPACE: Pause | A: Back | D: Forward | Q: Quit",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

    cv2.imshow("stack", imgStack)

    # ===== CONTROLS =====
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('d'):
        frame_pos = min(frame_pos + 5, total_frames - 1)
        paused = True
    elif key == ord('a'):
        frame_pos = max(frame_pos - 5, 0)
        paused = True

cap.release()
cv2.destroyAllWindows()

if lbw_detected:
    print("\nPotential LBW Detected!")
else:
    print("\nNo LBW Detected")
