"""
myFlaskApp.py defines a Flask web framework for sending/receiving data from demo.html, demo.css, and demo.js

Course: cs-396/398 Senior Projects
Course Coordinator: Professor Kenneth
Adviser: Professor Kenneth Arnold
Student: Joseph Jinn
Date: 1-20-20
############################################################################################################
Resources:

https://exploreflask.com/en/latest/organizing.html
(Flask)

https://healeycodes.com/javascript/python/beginners/webdev/2019/04/11/talking-between-languages.html
(Communication between Javascript and Python via Flask)
############################################################################################################
"""

import pandas as pd
from flask import Flask, jsonify, request, render_template
app = Flask(__name__, template_folder="templates", static_folder="static")

############################################################################################################


@app.route('/')
def home_page():
    """
    Serve and render our text prediction home.html page.
    Note: Need a default home page.
    :return: home.html - render template
    """
    return render_template('home.html')


@app.route('/demo_page')
def demo_page():
    """
    Serve and render our text prediction demo.html page.
    :return: demo.html - render template
    """
    return render_template('demo.html')


@app.route('/visualization_page')
def visualization_page():
    """
    Serve and render our text prediction visualization.html page.
    :return: visualization.html - render template
    """
    return render_template('visualization.html')

############################################################################################################


@app.route('/getInputText', methods=['GET', 'POST'])
def get_input_text():
    user_input_string = ''

    # POST request
    if request.method == 'POST':
        print(f'Incoming...')

        if request.is_json:
            print(f"User input text received")
            print(f"From HTML/Javascript: {request.get_json()}")  # parse as JSON
            request_json = request.get_json()
            user_input_string = request_json.get('user_input_text')
            print(f"User input text: {user_input_string}")
            return user_input_string, 200
        else:
            print(f"Data is not in JSON format!")
            return 'Failed to receive user input text.', 200

    # GET request
    else:
        message = {'flask_response_text': 'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers


@app.route('/sendVisualizationData', methods=['GET', 'POST'])
def send_visualization_data():
    user_input_string = ''

    # POST request
    if request.method == 'POST':
        print(f'Incoming...')

        if request.is_json:
            print(f"User input text received")
            print(f"From HTML/Javascript: {request.get_json()}")  # parse as JSON
            request_json = request.get_json()
            user_input_string = request_json.get('user_input_text')
            print(f"User input text: {user_input_string}")
            return user_input_string, 200
        else:
            print(f"Data is not in JSON format!")
            return 'Failed to receive user input text.', 200

    # GET request
    else:
        message = {'flask_response_text': 'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers

####################################################################################################################


if __name__ == "__main__":
    """
    Run the app on localhost.
    """
    app.run(host="127.0.0.1", port=8080, debug=True)

####################################################################################################################
