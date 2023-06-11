import numpy as np
from flask import Flask, render_template,Response,request
import cv2
import torch
import resnet_training

app = Flask(__name__)
camera = cv2.VideoCapture(0)
streaming = True
obj= resnet_training

@app.route("/")
def home():
    return render_template('index.html')

def generate_frames():
    global streaming
    while streaming:

        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_yolo_frames():
    model = torch.hub.load("yolov5", 'custom', r"C:\Users\rohin\Desktop\pythonProject\yolov5n.pt", source='local')
    while True:

        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            results=model(frame)

            ret, buffer = cv2.imencode('.jpg', np.squeeze(results.render()))
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route("/infy", methods = ['GET','POST'])
def run_infy():
    if request.method == 'POST':
        if 'camera-on' in request.form:
            global streaming
            streaming= True
            return render_template("infer.html")
        elif "yolo-on" in request.form:
            return render_template("infer1.html")

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_infer')
def video_infer():
    return Response(generate_yolo_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/imageclass")
def auto_dl():
    return render_template("IC.html")

@app.route("/training", methods= ['GET','POST'])
def train_dl():
    if request.method == 'POST':
        train_path = request.form['trainpath']
        test_path = request.form['testpath']
        epochs = request.form['epochs']
        lr = request.form['lr']
        print(train_path,test_path,epochs,lr)
        accuracy, train_loss = obj.givemodel().give_result(train_path, test_path,float(lr),int(epochs))
        return render_template("af_train.html" , accuracy= accuracy, train_loss=train_loss)





if __name__ == '__main__':
    app.run(debug=True)