#!/usr/bin/env python
"""
Real-Time Sign Language Detection and Broadcast via Flask-SocketIO

Modifications:
  - Removed signature handling to comply with Keras 3.x.
  - Removed TTS references.
  - Print only important messages (detections, or minimal debugging).
  - Lowered THRESHOLD from 0.5 to 0.3
  - Lowered CONSECUTIVE_NEEDED from 15 to 5
  - **Suppress all text except the name of the detected sign**
"""

import eventlet
eventlet.monkey_patch()

import os
import time
import base64
import threading
import shutil
import numpy as np
import cv2

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_socketio import SocketIO, join_room, emit

# --------------------- Configuration ---------------------
PORT = int(os.getenv('PORT', 8000))
SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')

app = Flask(__name__, template_folder='templates')
app.secret_key = SECRET_KEY

socketio = SocketIO(app, cors_allowed_origins="*")

# --------------------- Meeting & Model Globals ---------------------
# In-memory dictionary to store {meeting_id: password}
meetings = {}

# In-memory dict to track if a user is using sign language model: { socket_id: bool }
sign_language_enabled = {}

# In-memory dict to maintain per-user sign detection state
user_sign_states = {}

# (Optional) TURN Server Config
TURN_SERVER_URL = os.getenv('TURN_SERVER_URL', None)
TURN_SERVER_USERNAME = os.getenv('TURN_SERVER_USERNAME', '')
TURN_SERVER_CREDENTIAL = os.getenv('TURN_SERVER_CREDENTIAL','')

# --------------------- Training Data ---------------------
TRAINING_DATA_DIR = "training_data"
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
training_frame_buffers = {}  # {sid: {'sign': str, 'frames': list}}

# --------------------- Sign Language Model & Mediapipe Setup ---------------------
import tensorflow as tf
import mediapipe as mp

# Directly load the model without adding signatures
MODEL_PATH = "action.h5"

class _CompatInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['input_shape'] = kwargs.pop('batch_shape')[1:]
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['input_shape'] = config.pop('batch_shape')[1:]
        return super().from_config(config)

try:
    with tf.keras.utils.custom_object_scope({
        'InputLayer': _CompatInputLayer,
        'DTypePolicy': tf.keras.mixed_precision.Policy,
    }):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Sign language model loaded successfully.")
except Exception as e:
    print(f"Failed to load the sign language model: {e}")
    exit(1)

mp_holistic = mp.solutions.holistic

# Action classes — must match the exact order used when the model was trained.
# When you train a new model via /train, this list is updated automatically.
# To add more signs manually, record them via /train and retrain.
ACTIONS = [
    "george",
    "hello",
    "how",
    "not_signing",
    "you",
]

# Maps each sign name to the English text displayed on screen.
# - Leave "not_signing" as "" so idle frames produce no output.
# - All other signs should map to a word or phrase.
# - When a new model is trained via /train, add entries here to match.
SIGN_TO_ENGLISH = {
    # --- current trained signs ---
    "george":      "George",
    "hello":       "Hello",
    "how":         "How",
    "not_signing": "",        # intentionally silent — background/idle class
    "you":         "You",

    # --- common ASL signs (train these via /train to enable) ---
    "thank_you":   "Thank you",
    "yes":         "Yes",
    "no":          "No",
    "please":      "Please",
    "sorry":       "Sorry",
    "good":        "Good",
    "bad":         "Bad",
    "help":        "Help",
    "stop":        "Stop",
    "more":        "More",
    "again":       "Again",
    "name":        "Name",
    "what":        "What",
    "where":       "Where",
    "when":        "When",
    "why":         "Why",
    "want":        "Want",
    "like":        "Like",
    "love":        "Love",
    "friend":      "Friend",
    "family":      "Family",
    "home":        "Home",
    "school":      "School",
    "work":        "Work",
    "eat":         "Eat",
    "drink":       "Drink",
    "sleep":       "Sleep",
    "come":        "Come",
    "go":          "Go",
    "see":         "See",
    "hear":        "Hear",
    "understand":  "Understand",
    "know":        "Know",
    "think":       "Think",
    "feel":        "Feel",
    "fine":        "Fine",
    "bye":         "Goodbye",
    "morning":     "Good morning",
    "night":       "Good night",
    "i":           "I",
    "me":          "Me",
    "my":          "My",
    "we":          "We",
    "they":        "They",
    "he":          "He",
    "she":         "She",
    "it":          "It",
    "not":         "Not",
    "maybe":       "Maybe",
    "now":         "Now",
    "later":       "Later",
    "today":       "Today",
    "tomorrow":    "Tomorrow",
    "yesterday":   "Yesterday",
    "time":        "Time",
    "day":         "Day",
    "week":        "Week",
    "year":        "Year",
    "number":      "Number",
    "color":       "Color",
    "beautiful":   "Beautiful",
    "happy":       "Happy",
    "sad":         "Sad",
    "angry":       "Angry",
    "scared":      "Scared",
    "tired":       "Tired",
    "sick":        "Sick",
    "better":      "Better",
    "different":   "Different",
    "same":        "Same",
    "big":         "Big",
    "small":       "Small",
    "new":         "New",
    "old":         "Old",
    "open":        "Open",
    "close":       "Close",
    "give":        "Give",
    "take":        "Take",
    "make":        "Make",
    "buy":         "Buy",
    "money":       "Money",
    "water":       "Water",
    "food":        "Food",
    "book":        "Book",
    "car":         "Car",
    "house":       "House",
    "city":        "City",
    "country":     "Country",
    "hospital":    "Hospital",
    "doctor":      "Doctor",
    "police":      "Police",
    "emergency":   "Emergency",
}

# Constants for detection logic
TARGET_SEQUENCE_LEN = 10      # Number of frames per sequence
CONSECUTIVE_NEEDED = 5        # Reduced from 15 to 5
THRESHOLD = 0.3               # Reduced from 0.5 to 0.3

# --------------------- Helper Functions ---------------------
def extract_keypoints(results):
    """
    Flatten pose, face, left hand, right hand landmarks into a single vector.
    Must match the model's expected input size (e.g., 1662).
    """
    # Pose: 33 landmarks * 4 values = 132
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility]
                         for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(132)
    
    # Face: 468 landmarks * 3 = 1404
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z]
                         for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(1404)
    
    # Left hand: 21 landmarks * 3 = 63
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z]
                       for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)
    
    # Right hand: 21 landmarks * 3 = 63
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z]
                       for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)
    
    return np.concatenate([pose, face, lh, rh])

def sign_to_english_word(sign_name):
    """Convert recognized sign name to itself (no mapping needed)."""
    return SIGN_TO_ENGLISH.get(sign_name, sign_name)

# --------------------- Flask Routes ---------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        meeting_id = request.form.get('meeting_id')
        password = request.form.get('password')
        action = request.form.get('action')

        if not meeting_id or not password:
            flash('Meeting ID and password are required.')
            return redirect(url_for('index'))

        if action == 'Create':
            if meeting_id in meetings:
                flash('Meeting ID already exists. Choose another one.')
                return redirect(url_for('index'))
            meetings[meeting_id] = password
            session['meeting_id'] = meeting_id
            return redirect(url_for('room', meeting_id=meeting_id))

        elif action == 'Join':
            if meeting_id in meetings and meetings[meeting_id] == password:
                session['meeting_id'] = meeting_id
                return redirect(url_for('room', meeting_id=meeting_id))
            else:
                flash('Invalid Meeting ID or Password.')
                return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/room/<meeting_id>')
def room(meeting_id):
    if 'meeting_id' not in session:
        flash('Please join the meeting first.')
        return redirect(url_for('index'))
        
    if session['meeting_id'] != meeting_id:
        flash('Unauthorized access to the meeting room.')
        return redirect(url_for('index'))

    if meeting_id not in meetings:
        flash('This meeting no longer exists.')
        return redirect(url_for('index'))

    return render_template(
        'room.html',
        meeting_id=meeting_id,
        turn_url=TURN_SERVER_URL,
        turn_username=TURN_SERVER_USERNAME,
        turn_credential=TURN_SERVER_CREDENTIAL
    )

# --------------------- SocketIO Events ---------------------
@socketio.on('connect')
def handle_connect():
    sign_language_enabled[request.sid] = False
    # Initialize sign detection state for this user
    user_sign_states[request.sid] = {
        "sequence": [],
        "predictions": [],
        "last_recognized_sign": None
    }
    print(f"User connected: {request.sid}")

@socketio.on('join')
def handle_join(data):
    meeting_id = data.get('meeting_id')
    if not meeting_id:
        print("Join event received without meeting_id")
        return
    join_room(meeting_id)
    emit('user_joined',
         {'msg': 'A new user has joined the meeting.', 'timestamp': str(time.time())},  
         room=meeting_id)
    print(f"User {request.sid} joined meeting: {meeting_id}")

@socketio.on('signal')
def handle_signal(data):
    meeting_id = data.get('meeting_id')
    signal = data.get('signal')
    if not meeting_id or not signal:
        print("Invalid signal data received")
        return
    emit('signal', signal, room=meeting_id, include_self=False)

@socketio.on('subtitle_message')
def handle_subtitle_message(data):
    meeting_id = data.get('meeting_id')
    text = data.get('text', '')
    if not meeting_id or not text:
        print("subtitle_message received invalid data")
        return
    emit('subtitle_message', {'text': text}, room=meeting_id, include_self=False)

@socketio.on('toggle_sign_language')
def handle_toggle_sign_language(data):
    """
    Toggle the sign language model usage for this specific user (socket).
    data might look like: {"enabled": true, "meeting_id": "..."}
    """
    enabled = data.get('enabled', False)
    sign_language_enabled[request.sid] = enabled
    meeting_id = data.get('meeting_id', '')
    emit(
        'sign_language_status',
        {
            'user_sid': request.sid,
            'enabled': enabled
        },
        room=meeting_id,
        include_self=True
    )
    print(f"User {request.sid} toggled sign language model to {enabled}")

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Receives base64-encoded frames from the browser if sign language is toggled on.
    Runs Mediapipe + LSTM model => recognized_sign, then broadcasts the result.
    """
    meeting_id = data.get('meeting_id')
    if not meeting_id:
        print("No meeting_id provided in video_frame data.")
        return

    # If user hasn't toggled sign language, ignore frames
    if not sign_language_enabled.get(request.sid, False):
        return

    b64_data = data.get('data')
    if not b64_data:
        print("No data provided in video_frame.")
        return

    # Attempt to decode base64 -> OpenCV image
    try:
        header, encoded = b64_data.split(',', 1)
        decoded = base64.b64decode(encoded)
        np_data = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode frame.")
            return
    except Exception as e:
        print(f"Failed to decode base64 frame: {e}")
        return

    # Use Mediapipe to get keypoints
    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as holistic:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
    except Exception as e:
        print(f"Error processing frame with Mediapipe: {e}")
        return

    keypoints = extract_keypoints(results)
    state = user_sign_states.get(request.sid)
    if state is None:
        user_sign_states[request.sid] = {
            "sequence": [],
            "predictions": [],
            "last_recognized_sign": None
        }
        state = user_sign_states[request.sid]

    # Build a sequence of the last TARGET_SEQUENCE_LEN keypoint arrays
    seq = state["sequence"]
    seq.append(keypoints)
    seq = seq[-TARGET_SEQUENCE_LEN:]
    state["sequence"] = seq

    # Inference once we have enough frames
    if len(seq) == TARGET_SEQUENCE_LEN:
        print("Running model prediction...")
        try:
            res = model.predict(np.expand_dims(seq, axis=0))[0]
            print(f"Model prediction result: {res}")
        except Exception as e:
            print(f"Model prediction failed: {e}")
            return

        pred_class = np.argmax(res)
        confidence = res[pred_class]
        print(f"Predicted class: {pred_class}, Confidence: {confidence}")

        # Store predictions for "consecutive_needed" logic
        preds = state["predictions"]
        preds.append(pred_class)
        state["predictions"] = preds[-CONSECUTIVE_NEEDED:]

        if (
            len(state["predictions"]) >= CONSECUTIVE_NEEDED and
            len(set(state["predictions"][-CONSECUTIVE_NEEDED:])) == 1 and
            confidence > THRESHOLD
        ):
            recognized_sign = ACTIONS[pred_class]
            eng_word = sign_to_english_word(recognized_sign)

            # Only add if it's a new sign and not empty
            if eng_word and eng_word != state["last_recognized_sign"]:
                state["last_recognized_sign"] = eng_word

                # -- Only detection printout to terminal:
                print(f"Detected: {eng_word}")

                # Broadcast recognized sign to all in meeting
                emit('recognized_sign', {
                    'sign': eng_word,
                    'user_sid': request.sid
                }, room=meeting_id, include_self=True)

            # Clean up predictions to avoid repeated spam
            state["predictions"].clear()

@socketio.on('disconnect')
def handle_disconnect():
    sign_language_enabled.pop(request.sid, None)
    user_sign_states.pop(request.sid, None)
    print(f"User disconnected: {request.sid}")

# --------------------- Training Routes & Events ---------------------
@app.route('/train')
def train_page():
    return render_template('train.html')


@socketio.on('start_sequence')
def handle_start_sequence(data):
    sign = data.get('sign', '').strip().lower().replace(' ', '_')
    if not sign:
        emit('sequence_error', {'msg': 'Sign name is required'})
        return
    training_frame_buffers[request.sid] = {'sign': sign, 'frames': []}
    emit('sequence_started', {'sign': sign})


@socketio.on('training_frame')
def handle_training_frame(data):
    buf = training_frame_buffers.get(request.sid)
    if buf is None or len(buf['frames']) >= TARGET_SEQUENCE_LEN:
        return

    b64_data = data.get('data')
    if not b64_data:
        return

    try:
        _, encoded = b64_data.split(',', 1)
        decoded = base64.b64decode(encoded)
        np_data = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception:
        return

    try:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
        keypoints = extract_keypoints(results)
        buf['frames'].append(keypoints)

        frame_count = len(buf['frames'])
        emit('frame_processed', {'count': frame_count, 'total': TARGET_SEQUENCE_LEN})

        if frame_count == TARGET_SEQUENCE_LEN:
            sign = buf['sign']
            sign_dir = os.path.join(TRAINING_DATA_DIR, sign)
            os.makedirs(sign_dir, exist_ok=True)
            existing = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
            np.save(os.path.join(sign_dir, f'seq_{existing}.npy'), np.array(buf['frames']))
            del training_frame_buffers[request.sid]
            emit('sequence_saved', {'sign': sign, 'seq_num': existing + 1})
    except Exception as e:
        emit('sequence_error', {'msg': str(e)})


@socketio.on('get_training_data_info')
def handle_get_training_info(_data):
    info = {}
    if os.path.exists(TRAINING_DATA_DIR):
        for sign in sorted(os.listdir(TRAINING_DATA_DIR)):
            sign_dir = os.path.join(TRAINING_DATA_DIR, sign)
            if os.path.isdir(sign_dir):
                info[sign] = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
    emit('training_data_info', {'info': info})


@socketio.on('delete_sign_data')
def handle_delete_sign_data(data):
    sign = data.get('sign', '').strip()
    sign_dir = os.path.join(TRAINING_DATA_DIR, sign)
    if os.path.exists(sign_dir):
        shutil.rmtree(sign_dir)
    emit('sign_data_deleted', {'sign': sign})


@socketio.on('train_model')
def handle_train_model(data):
    epochs = int(data.get('epochs', 50))
    sid = request.sid

    def do_train():
        try:
            global model, ACTIONS

            signs = sorted([
                d for d in os.listdir(TRAINING_DATA_DIR)
                if os.path.isdir(os.path.join(TRAINING_DATA_DIR, d))
            ])

            if len(signs) < 2:
                socketio.emit('training_error', {'msg': 'Need at least 2 different signs to train.'}, room=sid)
                return

            socketio.emit('training_progress', {'msg': f'Loading sequences for {len(signs)} signs...', 'percent': 5}, room=sid)

            X, y = [], []
            for label_idx, sign in enumerate(signs):
                sign_dir = os.path.join(TRAINING_DATA_DIR, sign)
                for seq_file in sorted(f for f in os.listdir(sign_dir) if f.endswith('.npy')):
                    seq = np.load(os.path.join(sign_dir, seq_file))
                    if seq.shape == (TARGET_SEQUENCE_LEN, 1662):
                        X.append(seq)
                        y.append(label_idx)

            if len(X) == 0:
                socketio.emit('training_error', {'msg': 'No valid sequences found.'}, room=sid)
                return

            X = np.array(X)
            y = tf.keras.utils.to_categorical(y, num_classes=len(signs))

            socketio.emit('training_progress', {'msg': f'Loaded {len(X)} sequences. Building model...', 'percent': 20}, room=sid)

            new_model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, activation='relu',
                                     input_shape=(TARGET_SEQUENCE_LEN, 1662)),
                tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'),
                tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(len(signs), activation='softmax')
            ])
            new_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

            socketio.emit('training_progress', {'msg': 'Training...', 'percent': 25}, room=sid)

            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    percent = 25 + int(((epoch + 1) / epochs) * 70)
                    acc = logs.get('categorical_accuracy', 0)
                    socketio.emit('training_progress', {
                        'msg': f'Epoch {epoch + 1}/{epochs} — accuracy: {acc:.1%}',
                        'percent': percent
                    }, room=sid)

            new_model.fit(X, y, epochs=epochs, callbacks=[ProgressCallback()])

            socketio.emit('training_progress', {'msg': 'Saving model...', 'percent': 97}, room=sid)
            new_model.save(MODEL_PATH)
            model = new_model
            ACTIONS = signs

            socketio.emit('training_complete', {'msg': 'Model trained and saved!', 'actions': signs}, room=sid)

        except Exception as e:
            socketio.emit('training_error', {'msg': str(e)}, room=sid)

    threading.Thread(target=do_train, daemon=True).start()


# --------------------- Error Handlers ---------------------
@app.errorhandler(404)
def not_found_error(error):
    flash('The requested page was not found.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    flash('An internal server error occurred. Please try again later.')
    return redirect(url_for('index'))

# --------------------- Deployment on AWS ---------------------
def start_app():
    """
    Runs the Flask-SocketIO server.
    """
    print(f"Starting server on port {PORT}...")
    socketio.run(app, host='0.0.0.0', port=PORT)

if __name__ == '__main__':
    start_app()
