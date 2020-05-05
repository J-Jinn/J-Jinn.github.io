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
const debug = false;
const demo = {};
let saveJsonObject;

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
    demo.canvas.width = 3840;
    demo.canvas.height = 240;

    // Setup textarea for input.
    demo.textArea = $('#input-text-area')[0];
    demo.textArea.rows = 10;
    demo.textArea.cols = 250;

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
    console.log(output);
    demo.context.clearRect(0, 0, demo.canvas.width, demo.canvas.height);
    demo.context.fillStyle = "#27cd51";
    demo.context.font = "italic bold 12px/30px Georgia, serif";
    demo.context.fillText(output, 10, 50);

    let my_svg2 = d3.selectAll('svg#output-svg');
    my_svg2.selectAll("*").remove(); // Clear SVG.
    my_svg2.attr('width', 3840)
        .attr('height', 240)
        .style('background-color', '#181e3f');

    let my_g = my_svg2.selectAll('g')
        .data(output)
        .enter()
        .append('g');

    my_g.append('rect')
        .attr('width', 3840)
        .attr('height', 240)
        .attr('fill', '#39133f');

    my_g.selectAll('text')
        .data(output)
        .enter()
        .append('text')
        .attr("font-size", "10em")
        // .text(function(d) {return d})
        .text("hello, this is a test!")
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

    output = JSON.parse(output); // Derp, convert string to Javascript object first. (disable when testing)

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
        restructureData.push({"selected_token": key, "token_choices": tokenLists[key]})
    });
    if (debug) {
        console.log(restructureData);

        for (let i = 0; i < restructureData.length; i++) {
            console.log(`Restructured Data Word:\n ${restructureData[i].selected_token}`);
            console.log(`Restructured Data Tokens:\n ${restructureData[i].token_choices}`);
        }
    }

    let getSelectedText = "";
    let inputTokens = [];
    let inputString = "";

    // TODO - dynamic resizing of svg width based on the length of the predicted text and its tokens.
    let my_svg = d3.selectAll('svg#visualization-svg');  // select the svg element
    // https://stackoverflow.com/questions/10784018/how-can-i-remove-or-replace-svg-content
    my_svg.selectAll("*").remove(); // Permits re-draw for each iteration.
    my_svg.attr('width', 3840)
        .attr('height', 240)
        .selectAll('g')  // new selection starts here (and is empty for now)
        .data(restructureData)
        .enter()
        .append('g')     // selection now has n 'g' elements, one for each selected_token
        .style('transform', (d, i) => 'translate(' + (i * 100) + 'px, 50px)')
        .selectAll('text')
        .data(d => (d.token_choices || []).map(selected_token => ({
            selected_token: selected_token,
            matchesParent: selected_token === d.selected_token
        })))
        .enter()
        .append('text')
        .attr('x', 0)
        .attr('y', (d, i) => i * 20)
        .text(d => d.selected_token)
        .style('fill', d => d.matchesParent ? 'red' : '#00a515')
        .on('mouseover', function (d, i) {
            console.log("mouseover on", this);
            d3.select(this)
                .style('fill', '#ffaf53');
        })
        .on('mouseout', function (d, i) {
            console.log("mouseout", this);
            d3.select(this)
                .style('fill', d => d.matchesParent ? 'red' : '#00a515');
        })
        .on('click', function (d, i) {
            console.log("clicking on", this);
            getSelectedText = d3.select(this).text();
            console.log(getSelectedText);

            // Ghetto way of constructing the tokens we want to send back to the GPT-2 model.
            // https://stackoverflow.com/questions/1564818/how-to-break-nested-loops-in-javascript/1564838
            (function () {
                for (let key in tokenLists) {
                    console.log(`key: ${key}`);
                    for (let value = 0; value < tokenLists[key].length; value++) {
                        if (tokenLists[key][value] === getSelectedText) {
                            inputTokens.push(getSelectedText);
                            inputString = inputString.concat(getSelectedText);
                            return;
                        }
                    }
                    inputTokens.push(key);
                    // FIXME - not saving the first token...
                    inputString = inputString.concat(key);
                }
            })();
            console.log(`Input tokens: ${inputTokens}`);
            console.log(`Input string: ${inputString}`);

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
                    JSON.stringify({"user_input_text": inputString})
            }).then(function (response) {
                // Wait for the web server to return the results.
                return response.text();
            }).then(function (jsonObj) {
                // Output the returned data.
                console.log(`From Flask/Python: ${jsonObj}`);
                saveJsonObject = jsonObj;
            });
            // Clear for next selection and prediction.
            inputTokens = [];
            inputString = "";
            console.log(`saved json object: ${saveJsonObject}`);
            demo.drawVisualization(saveJsonObject);
        });
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////