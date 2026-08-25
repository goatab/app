# app.py
from flask import Flask

from infrastructure import load_hpo_data
from interface.routes import create_routes


def create_app():
    hpo_terms = load_hpo_data()
    app = Flask(__name__)
    app.register_blueprint(create_routes(hpo_terms))
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)