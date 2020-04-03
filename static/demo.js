/*
demo.js is the javascript file for demo.html

Course: CS-396/398 Senior Projects
Course Coordinator: Professor Kenneth
Adviser: Professor Kenneth Arnold
Student: Joseph Jinn
Date: 4-01-20
 */

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global variables.
const debug = true;
const demo = {};

const testData = {
    "data": [
        "Hello\nThe last time I saw a post on this blog was about my new blog, \"The Best",
        {
            "\n": [
                ",",
                ".",
                "\n"
            ],
            "The": [
                "\n",
                "I",
                "The"
            ],
            " last": [
                " first",
                " following",
                " last"
            ],
            " few": [
                " time",
                " thing",
                " few"
            ],
            " you": [
                " I",
                " we",
                " you"
            ],
            " wrote": [
                " checked",
                " saw",
                " wrote"
            ],
            " a": [
                " the",
                " this",
                " a"
            ],
            " post": [
                ",",
                ".",
                " post"
            ],
            " by": [
                " about",
                " on",
                " by"
            ],
            " reddit": [
                " this",
                " the",
                " reddit"
            ],
            " forum": [
                " blog",
                " site",
                " forum"
            ],
            " I": [
                ",",
                " was",
                " I"
            ],
            " about": [
                " in",
                " on",
                " about"
            ],
            " my": [
                " a",
                " the",
                " my"
            ],
            " wife": [
                " new",
                " first",
                " wife"
            ],
            " blog": [
                " book",
                " project",
                " blog"
            ],
            " and": [
                " The",
                " \"",
                " and"
            ],
            "What": [
                "The",
                "How",
                "What"
            ],
            " Best": [
                " Art",
                " New",
                " Best"
            ]
        }
    ]
};

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
    $('#generate-button').bind('click', demo.generate);

    // Setup canvas for output.
    demo.canvas = $('#output-canvas')[0];
    demo.context = demo.canvas.getContext('2d');
    demo.context.fillStyle = 'rgb(255,0,0)';

    // Test import data.
    // demo.importTestDataset();

    // Test draw visualization.
    // demo.drawVisualization(testData);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let inputText = '';

/**
 * Generate prediction and visualize results.
 */
demo.generate = () => {
    inputText = demo.getInputText();

    // POST the user input text to the web server.
    fetch('/getInputTextForVisualizationDemo', {
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
    }).then(function (jsonObj) {
        // Output the returned data.
        console.log(`From Flask/Python: ${jsonObj}`);
        demo.drawVisualization(jsonObj);
    });
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Get user input text.
 * @return string
 */
demo.getInputText = () => {
    return $('#input-text-area').val();
};

/**
 * Output text prediction results to canvas object.
 */
demo.outputResults = (output) => {
    demo.context.clearRect(0, 0, demo.canvas.width, demo.canvas.height);
    demo.context.fillStyle = "#27cd51;";
    demo.context.font = "italic bold 12px/30px Georgia, serif";
    demo.context.fillText(output, 10, 50);
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
    return undefined;
}

demo.importTestDataset = () => {
    /**
     * Function import the data.
     */
    // noinspection SpellCheckingInspection
    d3.csv("../static/files/next_token_logits_test.csv").then(function (data) {
        processData(data);
    }).catch(function (error) {
        console.log(error);
        console.log('Failed to import data.');
    });
    d3.selectAll('svg#visualization');
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Function to process and visualization GPT2-model predicted text.
 * Note: Ugly as hell code but functional...
 *
 * @param output - JSON response object.
 */
demo.drawVisualization = (output) => {

    // if (debug) {
    //     console.log(`Output variable:\n${output}`);
    //     console.log(`Output variable is of type: ${typeof output}`);
    //
    //     // Convert test data from GPT2-model into a JSON object as expected of Flask response.
    //     let jsonOutput = JSON.stringify(output);
    //     console.log(`JSON object:\n${jsonOutput}`);
    //     console.log(`JSON object type is: ${typeof jsonOutput}`);
    //
    //     // Convert JSON object into a Javascript object.
    //     let jsonOutputParsed = JSON.parse(jsonOutput);
    //     console.log(`JSON object parsed:\n${jsonOutputParsed}`);
    //     console.log(`JSON object parsed type is: ${typeof jsonOutputParsed}`);
    // }

    output = JSON.parse(output); // Derp, convert string to Javascript object first.

    // Separate the predicted text from its associated list of tokens for each word in the text.
    let predictedText = output["data"][0];
    let tokenLists = output["data"][1];
    let restructureData = [];
    // Take a look at our data.
    if (debug) {
        console.log(`Predicted text:\n${predictedText}`);

        Object.keys(tokenLists).forEach(function (key) {
            console.log(key + " " + tokenLists[key]);
        });
    }
    demo.outputResults(predictedText);
    // Convert data for use in D3.
    Object.keys(tokenLists).forEach(function (key) {
        restructureData.push({"word": key, "recs_shown": tokenLists[key]})
    });
    if (debug) {
        for (let i = 0; i < restructureData.length; i++) {
            console.log(`Restructured Data Word:\n ${restructureData[i].word}`);
            console.log(`Restructured Data Tokens:\n ${restructureData[i].recs_shown}`);
        }
    }

    // TODO - dynamic resizing of svg width based on the length of the predicted text and its tokens.
    d3.selectAll('svg#visualization-svg')  // select the svg element
        .attr('width', 3840)
        .attr('height', 240)
        .selectAll('g')  // new selection starts here (and is empty for now)
        .data(restructureData)
        .enter()
        .append('g')     // selection now has 11 rects, each associated with 1 row of data.
        .style('transform', (d, i) => 'translate(' + (i * 100) + 'px, 50px)')
        .selectAll('text')
        .data(d => (d.recs_shown || []).map(word => ({word: word, matchesParent: word === d.word})))
        .enter().append('text')
        .attr('x', 0)
        .attr('y', (d, i) => i * 20)
        .text(d => d.word)
        .style('fill', d => d.matchesParent ? 'red' : 'black')
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////