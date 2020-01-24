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
import csv
from flask import Flask, jsonify, request, render_template
# from huggingface_transformers import run_generation_visualization as rgv

app = Flask(__name__, template_folder="templates", static_folder="static")
debug = False


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
    """
    Function to POST for demo.js.
    TODO: rewrite run_generation_visualization in order to obtain actual data.
    :return: String/JSON
    """
    if request.method == 'POST':
        print(f'Incoming...')

        if request.is_json:
            request_json = request.get_json()
            user_input_string = request_json.get('user_input_text')

            # TODO: Here is where we need to run GPT-2 to produce predicted text to return.
            # rgv.main()

            if debug:
                print(f"User input text received")
                print(f"From HTML/Javascript: {request.get_json()}")  # parse as JSON
                print(f"User input text: {user_input_string}")
            return user_input_string, 200
        else:
            print(f"Data is not in JSON format!")
            return 'Failed to receive user input text.', 200


@app.route('/sendVisualizationData', methods=['GET', 'POST'])
def send_visualization_data():
    """
    Function to GET for visualization.js.
    TODO: rewrite run_generation_visualization in order to obtain actual data.
    :return: String/CSV
    """
    dataset_filepath = "huggingface_transformers/next_token_logits_test"

    df = pd.read_csv(f"{dataset_filepath}.csv", sep=',', encoding="utf-8")
    df.drop_duplicates(inplace=True)
    my_csv = df.to_csv(index=False, header=True, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    my_json = df.to_json(orient='records', lines=True)

    if debug:
        print(f"CSV conversion:\n{my_csv}")
        print(f"JSON conversion:\n{my_json}")

    return jsonify(my_json)  # serialize and use JSON headers


####################################################################################################################

if __name__ == "__main__":
    """
    Run the app on localhost.
    """
    app.run(host="127.0.0.1", port=8080, debug=True)

####################################################################################################################
