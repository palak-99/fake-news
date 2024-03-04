from flask import Flask
from routes import overlay_bp


def create_app():
    app = Flask(__name__, static_url_path='', static_folder='.')

    app.register_blueprint(overlay_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
