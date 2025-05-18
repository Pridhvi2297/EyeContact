import cv2
import dlib
import numpy as np

# Load Dlib Face Detector & Landmark Model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Camera Parameters (Assuming 640x480 Resolution)
FOCAL_LENGTH = 640
CENTER = (320, 240)  # Image Center
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, CENTER[0]],
    [0, FOCAL_LENGTH, CENTER[1]],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion

# 3D Model Points for Head Pose Estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose Tip
    (-30.0, -30.0, -30.0),  # Left Eye Corner
    (30.0, -30.0, -30.0),   # Right Eye Corner
    (-30.0, 30.0, -30.0),   # Left Mouth Corner
    (30.0, 30.0, -30.0),    # Right Mouth Corner
    (0.0, -60.0, -20.0)     # Chin
], dtype=np.float32)

def get_head_pose(landmarks):
    """Estimate head pose using facial landmarks."""
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose Tip
        (landmarks.part(36).x, landmarks.part(36).y),  # Left Eye Corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right Eye Corner
        (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth Corner
        (landmarks.part(54).x, landmarks.part(54).y),  # Right Mouth Corner
        (landmarks.part(8).x, landmarks.part(8).y)   # Chin
    ], dtype=np.float32)

    # Solve PnP (Estimate Rotation & Translation)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS
    )[:3]

    return success, rotation_vector, translation_vector

def stabilize_head(frame):
    """Align and stabilize head position."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get Head Pose
        success, rotation_vector, translation_vector = get_head_pose(landmarks)
        if not success:
            return frame

        # Compute Rotation Matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Define a fixed frontal-facing rotation matrix 
        fixed_rotation = np.eye(3, dtype=np.float32)

        # Compute the 3×4 Transformation Matrix
        new_pose_matrix = np.hstack((fixed_rotation, translation_vector.astype(np.float32)))

        # Extract 3×3 Homography Matrix (Fixed Shape Issue)
        homography_matrix = np.vstack([new_pose_matrix[:3], [0, 0, 1]])

        # Ensure the matrix is in the correct float format
        homography_matrix = homography_matrix.astype(np.float32)

        # Apply Warp Perspective (Natural Head Stabilization)
        stabilized_frame = cv2.warpPerspective(frame, homography_matrix, (frame.shape[1], frame.shape[0]))

        return stabilized_frame

    return frame  # If no face detected, return original frame

# Capture Video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    stabilized_frame = stabilize_head(frame)

    cv2.imshow("Head Pose Stabilization", stabilized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
