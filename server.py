from flask import Flask, render_template, Response, request, jsonify
from aiortc import RTCPeerConnection, RTCSessionDescription
import cv2
import json
import uuid
import asyncio
import logging
import time
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path if different

def generate_frames(path):
    # Load the YOLOv8 model

    # Open the video file
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        logging.error(f"Failed to open video file: {path}")
        return

    while True:
        start_time = time.time()
        success, frame = video.read()
        if not success:
            break
        else:
            # Process the frame using YOLOv8
            results = model(frame)[0]
            processed_frame = results.plot()  # Assuming render() returns list of processed frames

            # Encode the processed frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            # Yield the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            elapsed_time = time.time() - start_time
            logging.debug(f"Frame generation time: {elapsed_time} seconds")

    video.release()

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

from flask import redirect, url_for

# Route to video_feed directly
# @app.route('/')
# def index():
#     return redirect(url_for('video_feed'))


async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create and set the local description
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)



def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()


# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

@app.route('/video_feed')
def video_feed():
    path = request.args.get('path')
    return Response(generate_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')