import React, { useRef, useState } from "react";
import axios from "axios";

function App() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [emotion, setEmotion] = useState("");

    const startCamera = () => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoRef.current.srcObject = stream;
            })
            .catch(err => console.error("Error accessing webcam:", err));
    };

    const captureImage = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append("image", blob, "frame.jpg");

            try {
                const response = await axios.post("http://localhost:5000/predict", formData, {
                    headers: { "Content-Type": "multipart/form-data" }
                });
                setEmotion(response.data.emotion);
            } catch (error) {
                console.error("Error predicting emotion:", error);
            }
        }, "image/jpeg");
    };

    return (
        <div>
            <h1>Facial Expression Recognition</h1>
            <video ref={videoRef} autoPlay playsInline></video>
            <button onClick={startCamera}>Start Camera</button>
            <button onClick={captureImage}>Capture & Predict</button>
            <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
            {emotion && <h2>Detected Emotion: {emotion}</h2>}
        </div>
    );
}

export default App;