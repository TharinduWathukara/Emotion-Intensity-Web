import Emotion
from Emotion import PredictMultiEmotions, GetIntensity
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

# home
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/multiLabelEmotion', methods=['GET','POST'])
def multiLabelEmotion():
    if request.method == "POST":
        text = request.form['text']
        back_text = text
        print(text)

        emotions = PredictMultiEmotions(text)
        print(emotions)
        intensity = GetIntensity(text,emotions)
        print(intensity)

        # emotions = [1, 1, 0, 0, 0, 0]
        # intensity = [48.69553, 30.97951, 49.20415, 40.182337]

        # return render_template('multiLabelEmotion.html', text=back_text, multiLabel = [0.98176855, 0.98609114, 0.41127205, 0.10678381, 0.84794737, 0.52410958])
        return render_template('multiLabelEmotion.html', text=back_text, multiLabel=emotions, intensity=intensity)

    elif request.method == "GET":
        return render_template('multiLabelEmotion.html')

if __name__ == '__main__':
    # app.run(host='127.0.0.1')
    app.run(debug=True)