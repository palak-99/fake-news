from flask import Flask
from routes import overlay_bp
# from models import mongo
# from flask_cors import CORS


def create_app():
    app = Flask(__name__, static_url_path='', static_folder='.')
    # CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000/"}})

    # app.config['MONGO_URI'] = MONGO_URI

    # @app.route('/')
    # def index():
    #     return "Hello, World!"

    # @app.route('/', defaults={'path': ''})
    # @app.route('/<path:path>')
    # def serve(path):
    #     if path != "" and os.path.exists("build/" + path):
    #         return send_from_directory('build', path)
    #     else:
    #         return send_from_directory('build', 'index.html')

    app.register_blueprint(overlay_bp)
    # app.register_blueprint(livestream_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
