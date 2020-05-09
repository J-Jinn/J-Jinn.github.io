"""
myFlaskApp.py defines a Flask web framework for sending/receiving data from the demo website.

Course: CS-396/398 Senior Projects
Course Coordinator: Professor Kenneth
Adviser: Professor Kenneth Arnold
Student: Joseph Jinn
Date: 5-16-20
############################################################################################################
Resources:

https://exploreflask.com/en/latest/organizing.html
(Flask)
https://healeycodes.com/javascript/python/beginners/webdev/2019/04/11/talking-between-languages.html
(Communication between Javascript and Python via Flask)

Create requirements.txt:
    Go to your project environment conda activate <env_name>
    conda list gives you list of packages used for the environment.
    conda list -e > requirements.txt save all the info about packages to your folder.
    conda env export > <env_name>.yml.
    pip freeze.

Avoiding PythonAnywhere console closing issue due to excessive output:
https://www.pythonanywhere.com/forums/topic/20962/

"The solution is to redirect output to file: pip install -r requirements.txt > /tmp/temp"

FIXME: Need to include huggingface-transformers repository within GitHub Pages repository for Flask web app to function on PythonAnywhere.com
"""
############################################################################################################
import pandas as pd
import csv
import json
from flask import Flask, jsonify, request, render_template
from huggingface_transformers import run_generation_visualization_web_app as rgvwa

app = Flask(__name__, template_folder="templates", static_folder="static")
# app.config['JSON_SORT_KEYS'] = False
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


@app.route('/visualization_concept_page')
def visualization_concept_page():
    """
    Serve and render our text prediction visualization_concept.html page.
    :return: visualization_concept.html - render template
    """
    return render_template('visualization_concept.html')


############################################################################################################


@app.route('/sendVisualizationData', methods=['GET', 'POST'])
def send_visualization_data():
    """
    Function to GET for visualization_concept.js.
    :return: String/CSV
    """
    dataset_filepath = "static/files/next_token_logits_test"

    df = pd.read_csv(f"{dataset_filepath}.csv", sep=',', encoding="utf-8")
    df.drop_duplicates(inplace=True)
    my_csv = df.to_csv(index=False, header=True, quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    my_json = df.to_json(orient='records', lines=True)

    if debug:
        print(f"CSV conversion:\n{my_csv}")
        print(f"JSON conversion:\n{my_json}")

    return jsonify(my_json)  # serialize and use JSON headers


####################################################################################################################

@app.route('/getInputTextForVisualizationDemo', methods=['GET', 'POST'])
def get_input_text_for_visualization_demo():
    """
    Function to POST/GET for demo.js.
    :return: String/JSON
    """
    if request.method == 'POST':
        print(f'Incoming...')

        if request.is_json:
            request_json = request.get_json()
            user_input_string = request_json.get('user_input_text')

            # Call GPT-2 model, which returns predictions and other data.
            data = rgvwa.main(user_input_string)

            # # Faked return data for testing purposes.
            # data = ['Hello\nThe last time I saw a post on this blog was about my new blog, "The Best',
            #         {'\n': [',', '.', '\n'], 'The': ['\n', 'I', 'The'], ' last': [' first', ' following', ' last'],
            #          ' few': [' time', ' thing', ' few'], ' you': [' I', ' we', ' you'],
            #          ' wrote': [' checked', ' saw', ' wrote'], ' a': [' the', ' this', ' a'],
            #          ' post': [',', '.', ' post'], ' by': [' about', ' on', ' by'],
            #          ' reddit': [' this', ' the', ' reddit'], ' forum': [' blog', ' site', ' forum'],
            #          ' I': [',', ' was', ' I'], ' about': [' in', ' on', ' about'], ' my': [' a', ' the', ' my'],
            #          ' wife': [' new', ' first', ' wife'], ' blog': [' book', ' project', ' blog'],
            #          ' and': [' The', ' "', ' and'], 'What': ['The', 'How', 'What'],
            #          ' Best': [' Art', ' New', ' Best']}]

            if debug:
                print(f"User input text received")
                print(f"From HTML/Javascript: {request.get_json()}")  # parse as JSON
                print(f"User input text: {user_input_string}")
                print(f"Data from GPT-2 Model: {data}")

            return jsonify({'data': data}), 200
        else:
            print(f"Data is not in JSON format!")
            return 'Failed to receive user input text.', 200


####################################################################################################################

if __name__ == "__main__":
    """
    Run the app on localhost.
    """
    app.run(host="127.0.0.1", port=8080, debug=True)

####################################################################################################################
