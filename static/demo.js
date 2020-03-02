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
const testData = ['Hello.\n\n"We have to get the ball rolling and we\'re going to be ready for it', {0: [',', '.', '\n'], 1: ['\n', ' I', ' You'], 2: ['\n', 'I', 'The'], 3: ['I', 'The', '"'], 4: ['I', 'You', 'We'], 5: ["'re", ' are', ' have'], 6: [' a', ' to', ' been'], 7: [' be', ' do', ' get'], 8: [' the', ' this', ' out'], 9: [' ball', ' money', ' job'], 10: [' rolling', ' out', ' back'], 11: ['.', ',"', ' and'], 12: [' get', ' we', ' make'], 13: [' have', ' need', "'re"], 14: [' going', ' not', ' in'], 15: [' to', ' in', ' into'], 16: [' have', ' get', ' be'], 17: [' ready', ' able', ' a'], 18: [' to', ' for', '."'], 19: [' the', ' that', ' it']}];

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
    $('#dynamic-button').bind('click', demo.dynamic);

    // Setup canvas for output.
    demo.canvas = $('#output-canvas')[0];
    demo.context = demo.canvas.getContext('2d');
    demo.context.fillStyle = 'rgb(255,0,0)';

    // Test import data.
    demo.importTestDataset();

    // Test draw visualization.
    demo.drawVisualization(testData);
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

    if (debug) {
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
demo.dynamic = () => {
    programMode = "dynamic";
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
        console.log(`From Flask/Python: ${jsonObj}`);
        // Output the returned data.
        demo.outputResults(`${jsonObj}`);
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
    if (debug) {
        console.log(data);
    }
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

demo.drawVisualization = (output) => {

    // Convert test data from GPT2-model into a JSON object as expected of Flask response.
    let jsonOutput = JSON.stringify(output);
    console.log(`JSON object:\n${jsonOutput}`);
    console.log(`JSON object type is: ${typeof jsonOutput}`);

    // Convert JSON object into a Javascript object.
    let jsonOutputParsed = JSON.parse(jsonOutput);
    console.log(`JSON object parsed:\n${jsonOutputParsed}`);
    console.log(`JSON object parsed type is: ${typeof jsonOutputParsed}`);

    // Separate the predicted text from its associated list of tokens for each word in the text.
    let predictedText = jsonOutputParsed[0];
    let tokenLists = jsonOutputParsed[1];
    let restructureData = [];

    // Take a look at our data.
    if(debug) {
        console.log(`Predicted text:\n${predictedText}`);

        Object.keys(tokenLists).forEach(function(key) {
            console.log(key + " " + tokenLists[key]);
        });
    }

    // Convert data for use in D3.
    Object.keys(tokenLists).forEach(function(key) {
        restructureData.push({"word": key, "recs_shown": tokenLists[key]})
    });
    if(debug) {
        for(let i = 0; i < restructureData.length; i++) {
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
        .append('g')     // selection now has 11 rects, each associated with 1 row of data
        .style('transform', (d, i) => 'translate(' + (i * 100) + 'px, 50px)')
        .selectAll('text')
        .data(d => (d.recs_shown || []).map(word => ({word: word, matchesParent: word === d.word})))
        .enter().append('text')
        .attr('x', 0)
        .attr('y', (d, i) => i * 20)
        .text(d => d.word)
        .style('fill', d => d.matchesParent ? 'red' : 'black')

    // let suggestionsGroup = svg.append('g');
    //
    // suggestionsGroup.selectAll('g')
    //     .data(restructureData)
    //     .enter().append('g')
    //     .style('transform', (d, i) => 'translate(' + (i * 100) + 'px, 50px)')
    //     .selectAll('text')
    //     .data(d => (d.recs_shown || []).map(word => ({word: word, matchesParent: word === d.word})))
    //     .enter().append('text')
    //     .attr('x', 0)
    //     .attr('y', (d, i) => i * 20)
    //     .text(d => d.word)
    //     .style('fill', d => d.matchesParent ? 'red' : 'black')
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////