from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load your trained model
model = load_model("gesture_cnn_model.h5")

# Initial labels dictionary (we will fix after checking predictions)
labels = {
    0: "Palm",
    1: "L Sign",
    2: "Fist",
    3: "Fist Moved",
    4: "Thumb",
    5: "Index",
    6: "OK",
    7: "Palm Moved",
    8: "C Sign",
    9: "Down",
    10: "Unknown Gesture"  # Add this new one
}



# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)
    gesture_class = np.argmax(pred)

    # 🔍 Debugging prints (check terminal output)
    print("Prediction vector:", pred)
    print("Predicted class index:", gesture_class)

    # Map prediction to label
    gesture_label = labels.get(gesture_class, "Unknown")

    # Show result
    cv2.putText(frame, f"Gesture: {gesture_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
