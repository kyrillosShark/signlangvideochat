# Real-Time Sign Language Detection & WebRTC Video Chat

A Flask + Socket.IO application that provides:

1. **Live Video Chat** between participants using WebRTC.
2. **Real-Time Sign Language Detection** using TensorFlow + Mediapipe.
3. **Custom Model Training** — record your own signs and train a new model from the browser.
4. **Optional** text overlay and TTS (Text-to-Speech) on the client side.

---

## Table of Contents

1. [Features](#features)
2. [Demo Overview](#demo-overview)
3. [Setup & Installation](#setup--installation)
4. [Usage](#usage)
5. [Training Your Own Model](#training-your-own-model)
6. [Environment Variables](#environment-variables)
7. [Common Issues](#common-issues)
8. [Project Structure](#project-structure)
9. [License](#license)

---

## Features

- **WebRTC Video Chat**: Peer-to-peer audio/video streaming between participants in a meeting.
- **Pose & Hand Landmarks**: Mediapipe extracts pose, face, and hand keypoints from live camera frames.
- **LSTM Model**: A TensorFlow LSTM model interprets sequences of frames to detect sign language gestures.
- **Real-Time Feedback**: Detected signs are broadcast to all participants in the meeting room.
- **In-Browser Training**: Record video sequences for any sign directly from the browser and train a new LSTM model without leaving the app.
- **Optional TTS (Client-Side)**: Each user's browser can speak detected signs out loud.

---

## Demo Overview

1. **Create or Join a Meeting**: Enter a meeting ID and password to create or join a session.
2. **Join the Video Room**: Your local video appears immediately; the remote participant's video appears when they join.
3. **Enable Sign Language Detection**: Toggling it on sends your webcam frames to the server at ~25 FPS.
4. **Server Inference**: The server runs Mediapipe keypoint extraction + LSTM inference on each frame sequence.
5. **Broadcast Recognized Sign**: Detected signs are sent to everyone in the room and optionally spoken via TTS.

---

## Hardware Recommendations

### GPU (Strongly Recommended for Training)

Training the LSTM model on CPU is possible but slow. A **CUDA-capable NVIDIA GPU** dramatically reduces training time and is recommended if you plan to train your own signs.

| Hardware | Training speed | Notes |
|---|---|---|
| NVIDIA GPU (CUDA) | Fast | Recommended — enables GPU-accelerated TensorFlow |
| CPU only | Slow | Works, but expect several minutes per training run |

To enable GPU support:

1. Install the [NVIDIA CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) and [cuDNN 8.6](https://developer.nvidia.com/cudnn).
2. Install the GPU-enabled TensorFlow build:
   ```bash
   pip install tensorflow[and-cuda]==2.12.0
   ```
3. Verify TensorFlow detects your GPU:
   ```python
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   You should see your GPU listed. If the list is empty, check your CUDA/cuDNN versions match TensorFlow 2.12's requirements.

Real-time sign detection during video chat also benefits from a GPU, as Mediapipe keypoint extraction and model inference run faster.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kyrillosshark/videochat.git
cd videochat
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Current working dependency versions:

```
Flask==2.3.2
Flask-SocketIO==5.6.1
tensorflow==2.12.0
mediapipe==0.10.5
opencv-python==4.7.0.72
pyngrok==5.2.1
eventlet==0.33.3
protobuf==3.20.3
```

> **GPU users:** replace `tensorflow==2.12.0` with `tensorflow[and-cuda]==2.12.0` for CUDA support.

### 4. Ensure the Model File is Present

Place your trained TensorFlow model in the project root:

```
action.h5
```

The model path is set via `MODEL_PATH = "action.h5"` in `main.py`. If you don't have a pre-trained model, use the built-in training page to create one (see [Training Your Own Model](#training-your-own-model)).

---

## Usage

### 1. Run the Server

```bash
python main.py
```

The server starts at `http://localhost:8000` by default.

### 2. Open in Browser

- Navigate to `http://localhost:8000`.
- Create a meeting (e.g., ID: `room123`, Password: `abc`).
- Open a second browser tab or incognito window and join the same meeting.

### 3. Enable Camera & Microphone

Your browser will prompt for camera and mic access. Once granted, your local video appears. The remote participant's video appears when they connect.

### 4. Toolbar Controls

| Button | Description |
|---|---|
| Toggle Video | Enable/disable your camera |
| Toggle Audio | Enable/disable your microphone |
| Enable Sign Language | Start sending frames for sign detection |
| Enable TTS | Speak recognized signs aloud (local only) |
| Leave | End the call and return to home |

### 5. Sign Detection

When **Enable Sign Language** is active, the server detects signs and overlays the result on screen for all participants.

---

## Training Your Own Model

The app includes a built-in training interface at `http://localhost:8000/train`, accessible from the home page via the **Train Sign Language Model** button. No external tools or scripts are needed — everything runs in the browser and server.

### Training Workflow

#### Step 1 — Add Signs
Type a sign name (e.g. `hello`, `thank_you`, `not_signing`) and press Enter or click `+`. You need at least **2 different signs** to train a model.

#### Step 2 — Record Sequences
Select a sign from the list, then click **Record Sequence**:
- A **3-second countdown** gives you time to get into position.
- The server automatically captures **10 frames** from your webcam (roughly 2 seconds of motion) and extracts Mediapipe pose, face, and hand keypoints from each frame.
- A progress bar shows frame capture in real time.
- Each completed recording is saved as one sequence to disk.
- Repeat until you have enough sequences for each sign (30+ recommended).

While recording, a **live skeleton overlay** is drawn on the video — green for pose, red for left hand, blue for right hand, and teal for the face outline. Use this to confirm the model can see your landmarks clearly before recording.

#### Step 3 — Train the Model
Once you have data for at least 2 signs:
- Adjust the **Epochs** slider (50–100 is a good starting point).
- Click **Train Model**.
- Training runs on the server in a background thread. Live epoch-by-epoch progress and accuracy are displayed as training runs.
- When complete, the new `action.h5` is saved to disk and immediately loaded into the running server — no restart needed.
- The live meeting sign detection will now use your newly trained model.

### Model Architecture

The trained model is a 3-layer LSTM network:

```
Input: (10 frames × 1662 keypoints)
  → LSTM(64, return_sequences=True)
  → LSTM(128, return_sequences=True)
  → LSTM(64)
  → Dense(64) → Dense(32) → Dense(num_signs, softmax)
```

The 1662-dimensional keypoint vector per frame comes from:
- **Pose**: 33 landmarks × 4 values (x, y, z, visibility) = 132
- **Face**: 468 landmarks × 3 values (x, y, z) = 1404
- **Left hand**: 21 landmarks × 3 values = 63
- **Right hand**: 21 landmarks × 3 values = 63

### Tips for Good Accuracy

- Record **30 or more sequences** per sign for reliable detection.
- Vary your position, distance, and lighting between recordings to improve generalisation.
- Always include a **`not_signing`** class (idle hands, neutral position) — this prevents the model from hallucinating signs when you aren't signing.
- Keep hand movements **consistent** within each recording of the same sign.
- Use **50–100 epochs** for most datasets; smaller datasets may overfit above 100.
- A **CUDA GPU** (see [Hardware Recommendations](#hardware-recommendations)) will train 10–20× faster than CPU.

### Training Data Storage

Recorded sequences are saved to the `training_data/` directory and persist between server restarts:

```
training_data/
├── hello/
│   ├── seq_0.npy
│   ├── seq_1.npy
│   └── ...
├── not_signing/
│   └── ...
└── thank_you/
    └── ...
```

Each `.npy` file is a `(10, 1662)` NumPy array. You can add more sequences to any sign at any time and retrain — the training page loads existing data automatically. To remove a sign's data, click the trash icon next to it in the training interface.

---

## Environment Variables

Set these in your shell or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port the server listens on |
| `SECRET_KEY` | `your_secret_key_here` | Flask session secret key |
| `TURN_SERVER_URL` | *(none)* | TURN server URL for NAT traversal |
| `TURN_SERVER_USERNAME` | *(none)* | TURN server username |
| `TURN_SERVER_CREDENTIAL` | *(none)* | TURN server credential |

---

## Common Issues

**No Remote Video**
- Confirm both participants have joined the same meeting ID.
- Check browser console for ICE/SDP errors.
- On non-localhost networks, configure a TURN server via environment variables.

**Sign Detected but Not Displayed**
- The `SIGN_TO_ENGLISH` mapping in `main.py` may map some signs to `""`. Update the dictionary to return the sign name or a sentence.

**Model Not Found on Startup**
- Ensure `action.h5` exists in the project root, or use the training page to generate a new one.

**TensorFlow / Keras Version Errors**
- Use the exact versions listed in requirements.txt. The app includes compatibility patches for loading models saved with older Keras versions.

**TTS Not Heard by Other Participants**
- TTS runs locally in each participant's browser. It does not transmit over WebRTC. All participants receive the `recognized_sign` event and can individually enable TTS.

**Camera Access Denied on Training Page**
- Ensure you are on `http://localhost` or HTTPS. Browsers block camera access on plain HTTP for non-localhost origins.

---

## Project Structure

```
signlangvideochat/
│
├── main.py                 # Flask-SocketIO server, sign detection, training logic
├── requirements.txt        # Python dependencies
├── action.h5               # Trained TensorFlow LSTM model
│
├── templates/
│   ├── index.html          # Home page — create or join a meeting
│   ├── room.html           # Video chat room with sign language overlay
│   └── train.html          # In-browser model training interface
│
├── training_data/          # Recorded keypoint sequences (auto-created)
│   └── <sign_name>/
│       ├── seq_0.npy
│       └── ...
│
└── static/
    └── css/
        └── styles.css
```

---

## License

This project is licensed under the [MIT License](LICENSE), allowing you to freely modify and distribute it.
