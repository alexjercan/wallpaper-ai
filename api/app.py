from flask import Flask
from api.controller import simple_page

app = Flask(__name__)
app.register_blueprint(simple_page)