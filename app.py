from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
video = cv2.VideoCapture(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20), maxSize=(200, 200))

    filtered_faces = []
    for (x, y, w, h) in faces:
        is_overlapping = False
        for (xf, yf, wf, hf) in filtered_faces:
            if x > xf and y > yf and x + w < xf + wf and y + h < yf + hf:
                is_overlapping = True
                break
        if not is_overlapping:
            filtered_faces.append((x, y, w, h))

    faces_with_id = {}
    for face_id, face in enumerate(filtered_faces, start=1):
        x, y, w, h = face
        faces_with_id[face_id] = {
            'coordinates': face,
            'x': x,
            'y': y,
            'width': w,
            'height': h
        }

    return faces_with_id


def extract_face_video(frame, face):
    x, y, w, h = face['coordinates']
    face_frame = frame[y:y+h+100, x:x+w+20]
    return face_frame

def generate_frames():
    while True:
        success, frame = video.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_face_frames(face):
    while True:
        success, frame = video.read()
        if not success:
            break

        x, y, w, h = face['coordinates']
        cv2.rectangle(frame, (x, y), (x + w + 20, y + h + 100), (0, 255, 0), 2)
        face_frame = extract_face_video(frame, face)

        ret, buffer = cv2.imencode('.jpg', face_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    _, frame = video.read()
    faces = detect_faces(frame)
    num_faces = len(faces)
    return render_template('index.html', num_faces=num_faces, faces=faces)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_video/<int:face_id>/<int:x>/<int:y>/<int:width>/<int:height>')
def face_video(face_id, x, y, width, height):
    frame = video.read()[1]
    faces = detect_faces(frame)
    if face_id in faces:
        selected_face = faces[face_id]
        return Response(generate_face_frames(selected_face), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response('Invalid face ID')

if __name__ == '__main__':
    app.run(debug=True)
