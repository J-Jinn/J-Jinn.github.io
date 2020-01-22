/*
visualization.js is the javascript file for visualization.html

Course: cs-396/398 Senior Projects
Course Coordinator: Professor Kenneth
Adviser: Professor Kenneth Arnold
Student: Joseph Jinn
Date: 1-20-20
 */

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global variables.
const debug = true;
const visualization = {};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Call the init function once web page is ready.
 */
$(document).ready(function () {
    visualization.init();
});

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Initialization function.
 */
visualization.init = () => {
    // Setup canvas for output.
    visualization.canvas = $('#visualization-canvas')[0];
    visualization.context = visualization.canvas.getContext('2d');
    visualization.context.fillStyle = 'rgb(255,0,0)';

    // Load images.
    visualization.loadImages();
    // Draw images.
    visualization.drawBackground();
    // Import data.
    // visualization.importDataset();
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let background_image;

/**
 * Load background image.
 */
visualization.loadImages = () => {
    background_image = new Image();
    background_image.src = "../static/images/parchment_texture.jpg";
};

/**
 * Draw background.
 */
visualization.drawBackground = () => {
    visualization.context.clearRect(0, 0, visualization.canvas.width, visualization.canvas.height);
    visualization.context.drawImage(background_image, 0, 0,
        parseInt(visualization.canvas.width), parseInt(visualization.canvas.height));
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

/**
 * Function to import the data.
 */
visualization.importDataset = () => {
    // POST the user input text to the web server.
    fetch('/sendVisualizationData', {
        // Specify the method.
        method: 'POST',
        // Specify type of payload.
        headers: {
            'Content-Type': 'text/csv'
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
