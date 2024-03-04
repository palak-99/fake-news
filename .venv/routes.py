from flask import Blueprint, request, jsonify, Response, render_template, current_app

overlay_bp = Blueprint('overlay', __name__)


@overlay_bp.route('/')
def index():
    return render_template('index.html')


@overlay_bp.route('/news/<input_text>', methods=['POST'])
def newsVerify(input_text):
    # encoded = word_embedding(input_text)
    # padded_encoded_title = pad_sequences([encoded], maxlen=max_length, padding='pre')
    # output = fake_news_dectect_bot.predict(padded_encoded_title)
    # output = np.where(0.4 > output, 1, 0)
    output = 'Yes this News is fake'
    # if output[0][0] == 1:
    #     print('Yes this News is fake')
    # else:
    #     print('No, It is not fake')
    return jsonify({"res": output}), 200
