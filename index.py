import random
import operator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import expand_dims
from flask import Flask, render_template, Response, redirect, request, url_for, session, jsonify
from cv2 import cv2 as cv
from Camera import Camera

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'abcd$'
model_path = './model/'
model = load_model(model_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['player'] = request.form['playerName']
        session['turns'] = request.form['turns']
        session['prediction'] = ''
        return redirect(url_for('game'))
    return render_template('index.html')


@app.route('/game')
def game():
    player_name = session['player']
    turns = session['turns']
    rnd_num = generate_rnd_num(int(turns))
    return render_template(
        'game.html',
        player_name=player_name,
        turns=turns,
        rnd_num=rnd_num,
    )


@app.route('/prediction/<round>', methods=['GET', 'POST'])
def get_prediction(round):
    frame = Camera()
    frame.take_picture(round)
    photo = cv.imread('./choices/{}.jpeg'.format(str(round)))
    pred_frame = cv.resize(photo, (300, 300))
    pred_frame = img_to_array(pred_frame, dtype='float32')
    prediction = model.predict(expand_dims(pred_frame,  axis=0))
    prediction = prediction[0]
    predictions = {}
    predictions['Rock!'] = prediction[0]
    predictions['Paper!'] = prediction[1]
    predictions['Scissors!'] = prediction[2]
    prediction = max(
        predictions.items(), key=operator.itemgetter(1))[0]
    print(jsonify(prediction))
    return jsonify(prediction)


# Runs camera
def get_camera(camera):
    while True:
        frame = camera.get_video_capture()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_page')
def video():
    return Response(get_camera(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/end')
def end_game():
    session.clear()
    return redirect(url_for('index'))

### Utilities ###

def generate_rnd_num(num):
    numbers = list()
    for i in range(0, num+1):
        numbers.append(random.randint(0, 2))
    return numbers


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
