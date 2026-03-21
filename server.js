// Common scripts for all pages

// Check browser compatibility (for index.html)
if (document.getElementById('browserCheck')) {
    document.addEventListener('DOMContentLoaded', function() {
      const browserCheck = document.getElementById('browserCheck');
      browserCheck.style.display = 'block';

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        browserCheck.className = 'alert alert-danger';
        browserCheck.textContent = 'Your browser does not support webcam access. Please use a modern browser.';
      } else {
        browserCheck.className = 'alert alert-success';
        browserCheck.textContent = 'Browser supports webcam access!';
        setTimeout(() => browserCheck.style.display = 'none', 3000);
      }
    });
  }

// Form validation for index.html
if (document.querySelector('form')) {
    document.addEventListener('DOMContentLoaded', function () {
      const form = document.querySelector('form');

      // Form Validation
      form.addEventListener('submit', function (event) {
        if (!form.checkValidity()) {
          event.preventDefault();
          event.stopPropagation();
        }
        form.classList.add('was-validated');
      }, false);
    });
}

// Scripts specific to room.html
if (document.getElementById('topToolbar')) {
    /*****************************************************
     * GLOBALS & DOM REFERENCES
     *****************************************************/
    const MEETING_ID = "{{ meeting_id }}";
    const statusIndicator = document.getElementById("statusIndicator");
    const errorMessages = document.getElementById("errorMessages");
    const signOverlay = document.getElementById("signOverlay");
    const sentenceOverlay = document.getElementById("sentenceOverlay");

    const localVideo = document.getElementById("localVideo");
    const remoteVideo = document.getElementById("remoteVideo");
    const remotePlaceholder = document.getElementById("remotePlaceholder"); // Added
    const keypointCanvas = document.getElementById("keypointCanvas");
    const keypointCtx = keypointCanvas.getContext("2d");

    // WebRTC + Socket
    const socket = io();
    let localStream;
    let peerConnection;
    let posenetModel;

    // Flags
    let usingSignLanguage = false;
    let frameTimer = null;
    let ttsEnabled = false;

    // Selected male voice
    let selectedMaleVoice = null;

    /*****************************************************
     * ERROR + STATUS HELPERS
     *****************************************************/
    function showError(message, timeout = 5000) {
      console.error(message);
      errorMessages.textContent = message;
      errorMessages.style.display = "block";
      if (timeout) {
        setTimeout(() => {
          errorMessages.style.display = "none";
        }, timeout);
      }
    }

    function updateStatus(msg) {
      console.log(msg);
      statusIndicator.textContent = `Status: ${msg}`;
    }

    /*****************************************************
     * MEDIA + PoseNet SETUP
     *****************************************************/
    async function setupLocalMedia() {
      try {
        updateStatus("Requesting camera/mic access...");
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localVideo.srcObject = localStream;
        updateStatus("Local media connected");

        socket.emit("join", { meeting_id: MEETING_ID });

        // Wait a moment for video to have a size
        localVideo.onloadedmetadata = async () => {
          console.log("Local video metadata loaded.", localVideo.videoWidth, localVideo.videoHeight);
          await loadPoseNet();
        };
      } catch (err) {
        showError(`Media access error: ${err.message}`, 0);
      }
    }

    async function loadPoseNet() {
      try {
        updateStatus("Loading PoseNet...");
        posenetModel = await posenet.load({
          architecture: "MobileNetV1",
          outputStride: 16,
          inputResolution: { width: 640, height: 480 },
          multiplier: 0.75
        });
        updateStatus("PoseNet loaded successfully");
        startPoseDetectionLoop();
      } catch (err) {
        showError(`PoseNet loading error: ${err.message}`, 0);
      }
    }

    function startPoseDetectionLoop() {
      // Resize canvas to match actual video resolution
      keypointCanvas.width = localVideo.videoWidth;
      keypointCanvas.height = localVideo.videoHeight;
      keypointCanvas.style.width = `${localVideo.clientWidth}px`;
      keypointCanvas.style.height = `${localVideo.clientHeight}px`;
      console.log("Canvas:", keypointCanvas.width, keypointCanvas.height);

      async function poseDetectFrame() {
        if (localVideo.readyState === localVideo.HAVE_ENOUGH_DATA && posenetModel) {
          const pose = await posenetModel.estimateSinglePose(localVideo, {
            flipHorizontal: true
          });

          keypointCtx.clearRect(0, 0, keypointCanvas.width, keypointCanvas.height);
          drawKeypoints(pose.keypoints, 0.5);
          drawSkeleton(pose.keypoints, 0.5);
        }
        requestAnimationFrame(poseDetectFrame);
      }
      poseDetectFrame();
    }

    function drawKeypoints(keypoints, minConfidence) {
      keypoints.forEach(({ position: { x, y }, score }) => {
        if (score >= minConfidence) {
          keypointCtx.beginPath();
          keypointCtx.arc(x, y, 5, 0, 2 * Math.PI);
          keypointCtx.fillStyle = "aqua";
          keypointCtx.fill();
        }
      });
    }

    function drawSkeleton(keypoints, minConfidence) {
      const adjacentKeyPoints = posenet.getAdjacentKeyPoints(keypoints, minConfidence);
      adjacentKeyPoints.forEach(([from, to]) => {
        keypointCtx.beginPath();
        keypointCtx.moveTo(from.position.x, from.position.y);
        keypointCtx.lineTo(to.position.x, to.position.y);
        keypointCtx.strokeStyle = "lime";
        keypointCtx.lineWidth = 2;
        keypointCtx.stroke();
      });
    }

    /*****************************************************
     * SENDING FRAMES TO SERVER (VIDEO_FRAME EVENT)
     *****************************************************/
    function startSendingFramesToServer() {
      const FPS = 25;
      frameTimer = setInterval(() => {
        if (!usingSignLanguage) return;
        if (!localVideo || localVideo.readyState < 2) return;

        const offscreen = document.createElement("canvas");
        offscreen.width = 320;
        offscreen.height = 240;
        const ctx = offscreen.getContext("2d");
        ctx.drawImage(localVideo, 0, 0, offscreen.width, offscreen.height);

        const base64Data = offscreen.toDataURL("image/jpeg", 0.5);
        socket.emit("video_frame", {
          data: base64Data,
          meeting_id: MEETING_ID
        });
      }, 1000 / FPS);
      console.log("Started sending frames to server.");
    }

    function stopSendingFramesToServer() {
      if (frameTimer) clearInterval(frameTimer);
      frameTimer = null;
      console.log("Stopped sending frames to server.");
    }

    /*****************************************************
     * WEBRTC PEER CONNECTION
     *****************************************************/
    const iceConfig = {
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" }
      ]
    };

    function createPeerConnection() {
      try {
        peerConnection = new RTCPeerConnection(iceConfig);

        peerConnection.onicecandidate = (e) => {
          if (e.candidate) {
            socket.emit("signal", {
              meeting_id: MEETING_ID,
              signal: { candidate: e.candidate }
            });
          }
        };

        peerConnection.oniceconnectionstatechange = () => {
          const state = peerConnection.iceConnectionState;
          updateStatus(`ICE Connection: ${state}`);
          if (["disconnected", "failed", "closed"].includes(state)) {
            remoteVideo.srcObject = null;
            showRemotePlaceholder(); // Show placeholder when disconnected
            if (state === "failed") {
              showError("Connection failed. Try refreshing the page.");
            }
          }
        };

        peerConnection.ontrack = (event) => {
          if (event.streams && event.streams[0]) {
            remoteVideo.srcObject = event.streams[0];
            updateStatus("Remote video connected");
            // Hide placeholder when remote video is connected
            hideRemotePlaceholder();
          }
        };

        return true;
      } catch (err) {
        showError(`Failed to create peer connection: ${err.message}`);
        return false;
      }
    }

    /*****************************************************
     * SOCKET.IO HANDLERS
     *****************************************************/
    socket.on("connect", () => {
      updateStatus("Socket Connected");
    });

    socket.on("connect_error", (err) => {
      showError(`Socket connection error: ${err.message}`);
    });

    socket.on("user_joined", (data) => {
      console.log("User joined:", data);
      updateStatus("Peer joined - Creating connection...");
      if (!peerConnection && createPeerConnection()) {
        localStream.getTracks().forEach((track) => {
          peerConnection.addTrack(track, localStream);
        });
        peerConnection
          .createOffer()
          .then((offer) => peerConnection.setLocalDescription(offer))
          .then(() => {
            socket.emit("signal", {
              meeting_id: MEETING_ID,
              signal: { sdp: peerConnection.localDescription }
            });
          })
          .catch((err) => {
            showError(`Failed to create offer: ${err.message}`);
          });
      }
    });

    socket.on("signal", async (data) => {
      try {
        if (data.sdp) {
          await handleSDP(data.sdp);
        } else if (data.candidate) {
          await handleCandidate(data.candidate);
        }
      } catch (err) {
        showError(`Signaling error: ${err.message}`);
      }
    });

    async function handleSDP(sdp) {
      if (!peerConnection) {
        if (!createPeerConnection()) return;
        localStream.getTracks().forEach((track) => {
          peerConnection.addTrack(track, localStream);
        });
      }
      await peerConnection.setRemoteDescription(new RTCSessionDescription(sdp));
      if (sdp.type === "offer") {
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        socket.emit("signal", {
          meeting_id: MEETING_ID,
          signal: { sdp: peerConnection.localDescription }
        });
      }
    }

    async function handleCandidate(candidate) {
      if (peerConnection) {
        await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
      }
    }

    // Recognized signs from the server
    socket.on("recognized_sign", (data) => {
      // Example: data => { sign: "...", sentence: "...", user_sid: "..." }
      console.log("recognized_sign:", data);
      if (data.sign) {
        signOverlay.textContent = data.sign;
        signOverlay.style.display = "block";
        setTimeout(() => {
          signOverlay.style.display = "none";
        }, 3000);
      }
      if (data.sentence) {
        sentenceOverlay.textContent = data.sentence;
        sentenceOverlay.style.display = "block";
        setTimeout(() => {
          sentenceOverlay.style.display = "none";
        }, 3000);
      }
      if (ttsEnabled) {
        if (data.sentence) speakText(data.sentence);
        else if (data.sign) speakText(data.sign);
      }
    });

    socket.on("sign_language_status", (data) => {
      console.log("sign_language_status:", data);
    });

    /*****************************************************
     * BUTTON HANDLERS
     *****************************************************/
    document.getElementById("leaveBtn").addEventListener("click", () => {
      if (localStream) localStream.getTracks().forEach((track) => track.stop());
      if (peerConnection) peerConnection.close();
      socket.disconnect();
      window.location.href = "/";
    });

    document.getElementById("toggleVideo").addEventListener("click", () => {
      const videoTrack = localStream?.getVideoTracks?.()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        const btn = document.getElementById("toggleVideo");
        btn.textContent = videoTrack.enabled ? "Disable Video" : "Enable Video";
      }
    });

    document.getElementById("toggleAudio").addEventListener("click", () => {
      const audioTrack = localStream?.getAudioTracks?.()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        const btn = document.getElementById("toggleAudio");
        btn.textContent = audioTrack.enabled ? "Disable Audio" : "Enable Audio";
      }
    });

    document.getElementById("toggleSignLanguageBtn").addEventListener("click", () => {
      usingSignLanguage = !usingSignLanguage;
      socket.emit("toggle_sign_language", {
        enabled: usingSignLanguage,
        meeting_id: MEETING_ID
      });
      const btn = document.getElementById("toggleSignLanguageBtn");
      btn.textContent = usingSignLanguage ? "Disable Sign Language" : "Enable Sign Language";

      if (usingSignLanguage) {
        startSendingFramesToServer();
      } else {
        stopSendingFramesToServer();
        signOverlay.style.display = "none";
        sentenceOverlay.style.display = "none";
      }
    });

    const toggleTTSBtn = document.getElementById("toggleTTSBtn");
    toggleTTSBtn.addEventListener("click", () => {
      ttsEnabled = !ttsEnabled;
      toggleTTSBtn.textContent = ttsEnabled ? "Disable TTS" : "Enable TTS";
      if (ttsEnabled) speakText("Text to speech is now enabled.");
    });

    /*****************************************************
     * BROWSER SPEECH SYNTHESIS WITH MALE VOICE
     *****************************************************/
    function logAvailableVoices() {
      const voices = speechSynthesis.getVoices();
      console.log("Available Voices:");
      voices.forEach((voice, index) => {
        console.log(`${index + 1}. Name: ${voice.name}, Lang: ${voice.lang}`);
      });
    }

    function selectMaleVoice() {
      const voices = speechSynthesis.getVoices();

      // List of known male voice names (adjust based on your console logs)
      const maleVoiceNames = [
        'Google US English', // Common in Chrome
        'Microsoft David Desktop - English (United States)', // Common in Windows
        'Alex', // macOS
        // Add more male voice names as needed
      ];

      // Find the first matching male voice
      selectedMaleVoice = voices.find(voice => maleVoiceNames.includes(voice.name));

      // Fallback if no male voice is found
      if (!selectedMaleVoice && voices.length > 0) {
        selectedMaleVoice = voices[0];
        console.warn('No male voice found. Falling back to the default voice:', selectedMaleVoice.name);
      } else if (!selectedMaleVoice) {
        console.error('No voices available for speech synthesis.');
      }
    }

    // Initialize voice selection
    if (speechSynthesis.onvoiceschanged !== undefined) {
      speechSynthesis.onvoiceschanged = () => {
        selectMaleVoice();
        logAvailableVoices(); // Optional: Log voices after selection
      };
    } else {
      selectMaleVoice();
      logAvailableVoices(); // Optional: Log voices after selection
    }

    // Modified speakText function using the selected male voice
    function speakText(text) {
      if ("speechSynthesis" in window) {
        if (speechSynthesis.speaking) {
          console.log("Speech synthesis already in progress. Queuing the utterance.");
        }

        console.log("Speaking:", text);
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "en-US";

        if (selectedMaleVoice) {
          utterance.voice = selectedMaleVoice;
        } else {
          console.warn('No male voice selected. Using the default voice.');
        }

        utterance.pitch = 1; // Adjusted from 0.05 to 1 for better audibility
        utterance.rate = 1;  // Adjust rate as needed (0.1 to 10)

        speechSynthesis.speak(utterance);
      } else {
        console.warn("speechSynthesis not supported in this browser.");
      }
    }

    /*****************************************************
     * REMOTE VIDEO PLACEHOLDER HANDLERS
     *****************************************************/
    function showRemotePlaceholder() {
      remotePlaceholder.style.display = "flex";
    }

    function hideRemotePlaceholder() {
      remotePlaceholder.style.display = "none";
    }

    // Event listeners to toggle placeholder based on remote video state
    remoteVideo.addEventListener('playing', () => {
      hideRemotePlaceholder();
    });

    remoteVideo.addEventListener('pause', () => {
      showRemotePlaceholder();
    });

    remoteVideo.addEventListener('ended', () => {
      showRemotePlaceholder();
    });

    remoteVideo.addEventListener('error', () => {
      showRemotePlaceholder();
    });

    // Initially show placeholder if no remote stream is present
    socket.on("disconnect", () => {
      showRemotePlaceholder();
    });

    /*****************************************************
     * SWAP VIDEO WINDOWS ON DOUBLE-CLICK
     *****************************************************/
    function setupSwapOnDoubleClick() {
      const remoteContainer = document.getElementById("remoteContainer");
      const localContainer = document.getElementById("localContainer");

      // Function to toggle the 'swapped' class on the body
      function toggleSwap() {
        document.body.classList.toggle("swapped");
      }

      // Add double-click event listeners to both containers
      remoteContainer.addEventListener("dblclick", toggleSwap);
      localContainer.addEventListener("dblclick", toggleSwap);
    }

    /*****************************************************
     * INIT
     *****************************************************/
    setupLocalMedia();
    setupSwapOnDoubleClick(); // Initialize swap functionality
}
  