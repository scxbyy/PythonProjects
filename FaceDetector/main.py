import cv2
import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Set up camera
cap = cv2.VideoCapture(0)

# Set the desired resolution (width, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up Pygame display
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Face Detection")

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Find the largest face
    if len(faces) > 0:
        # Select the largest face based on area
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # (x, y, w, h)
        x, y, w, h = largest_face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Convert frame to RGB format for Pygame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)  # Rotate to fit Pygame's display
    frame_surface = pygame.surfarray.make_surface(frame)

    # Blit the frame to the Pygame screen
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

# Cleanup
cap.release()
pygame.quit()
