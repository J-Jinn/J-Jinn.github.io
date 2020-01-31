/*
demo.js is the javascript file for demo.html

Course: cs-396/398 Senior Projects
Course Coordinator: Professor Kenneth
Adviser: Professor Kenneth Arnold
Student: Joseph Jinn
Date: 1-20-20
 */

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global variables.
const debug = true;
const demo = {};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Call the init function once web page is ready.
 */
$(document).ready(function () {
    demo.init();
});

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Initialization function.
 */
demo.init = () => {
    // Bind functions to event handlers.
    $('#automatic-button').bind('click', demo.auto);
    $('#manual-button').bind('click', demo.manual);

    // Setup canvas for output.
    demo.canvas = $('#output-canvas')[0];
    demo.context = demo.canvas.getContext('2d');
    demo.context.fillStyle = 'rgb(255,0,0)';

    // Test import data.
    demo.importTestDataset();
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let programMode = 'auto';
let inputText = '';

/**
 * Automatic mode.
 */
demo.auto = () => {
    programMode = "auto";
    inputText = demo.getInputText();

    if(debug) {
        // Test we can convert to JSON format.
        let test_string_conversion = JSON.stringify({"user_input_text": inputText});
        console.log(`Conversion to string: ${test_string_conversion}`);
        console.log(`Type is: ${typeof test_string_conversion}`);
        let test_json_conversion = JSON.parse(test_string_conversion);
        console.log(`Conversion to JSON: ${test_json_conversion}`);
        console.log(`Type is: ${typeof test_json_conversion}`);
    }

    // POST the user input text to the web server.
    fetch('/getInputText', {
        // Specify the method.
        method: 'POST',
        // Specify type of payload.
        headers: {
            'Content-Type': 'application/json'
        },
        // A JSON payload.
        body:
            JSON.stringify({"user_input_text": inputText})
    }).then(function (response) {
        // Wait for the web server to return the results.
        return response.text();
    }).then(function (text) {
        console.log(`From Flask/Python: ${text}`);
        // Output the results of the text prediction model.
        demo.outputResults(`${text}`)
    });
};

/**
 * Manual mode.
 */
demo.manual = () => {
    programMode = "manual";
    inputText = demo.getInputText();
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Get user input text.
 * @return string
 */
demo.getInputText= () => {
    return $('#input-text-area').val();
};

/**
 * Output text prediction results to canvas object.
 */
demo.outputResults = (output) => {
    demo.context.clearRect(0, 0, demo.canvas.width, demo.canvas.height);
    demo.context.fillStyle = "#36139b";
    demo.context.font = "italic bold 12px/30px Georgia, serif";
    demo.context.fillText( output, 10, 50);
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Function to process imported data.
 *
 * @param data
 * @returns {undefined}
 */
function processData(data) {
    // Do something.
    if(debug) {
        console.log(data);
    }
    return undefined;
}

demo.importTestDataset = () => {
    /**
     * Function import the data.
     */
    // noinspection SpellCheckingInspection
    d3.csv("../J-Jinn.github.io/huggingface_transformers/next_token_logits_test.csv").then(function(data) {
        processData(data);
    }).catch(function(error) {
        console.log(error);
        console.log('Failed to import data.');
    });
    d3.selectAll('svg#visualization');
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
