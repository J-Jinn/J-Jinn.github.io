/*
visualization_concept.js is the javascript file for visualization_concept.html

Course: cs-396/398 Senior Projects
Course Coordinator: Professor Kenneth
Adviser: Professor Kenneth Arnold
Student: Joseph Jinn
Date: 1-20-20
 */

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global variables.
const debug = true;
const visualization_concept = {};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Call the init function once web page is ready.
 */
$(document).ready(function () {
    visualization_concept.init();
});

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Initialization function.
 */
visualization_concept.init = () => {
    // Bind functions to event handlers.
    $('#generate-button').bind('click', visualization_concept.generate);

    // Setup canvas for output.
    visualization_concept.canvas = $('#visualization-canvas')[0];
    visualization_concept.context = visualization_concept.canvas.getContext('2d');
    visualization_concept.context.fillStyle = 'rgb(255,0,0)';

    // Load images.
    visualization_concept.loadImages();
    // Draw images.
    visualization_concept.drawBackground();
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let background_image;

/**
 * Load background image.
 */
visualization_concept.loadImages = () => {
    background_image = new Image();
    background_image.src = "../static/images/parchment_texture.jpg";
};

/**
 * Draw background.
 */
visualization_concept.drawBackground = () => {
    visualization_concept.context.clearRect(0, 0, visualization_concept.canvas.width, visualization_concept.canvas.height);
    visualization_concept.context.drawImage(background_image, 0, 0,
        parseInt(visualization_concept.canvas.width), parseInt(visualization_concept.canvas.height));
};

/**
 * Generate visualization_concept button.
 */
visualization_concept.generate = () => {
    // Test import data as JSON.
    visualization_concept.importDatasetAsJson();
    // Test import CSV using D3.
    visualization_concept.importDatasetWithD3();

    // Draw the word tree.
    visualization_concept.drawWordTree();
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Function to process imported data.
 *
 * @param data
 */
function processData(data) {
    // Do something.
    if (debug) {
        console.log("Imported data successfully!");
        console.log(data);
    }
}

/**
 * Function to import the data as CSV using D3.
 * Note: Must serve files from static folder or sub-folders when using Flask web server.
 */
visualization_concept.importDatasetWithD3 = () => {
    // noinspection SpellCheckingInspection
    d3.csv("/static/files/next_token_logits_test.csv").then(function (data) {
        processData(data);
    }).catch(function (error) {
        console.log(error);
        console.log('Failed to import data.');
    });
    d3.selectAll('svg#visualization-svg');
};

/**
 * Function to import the data as JSON from Flask web server using GET.
 */
visualization_concept.importDatasetAsJson = () => {
    // GET is the default method, so we don't need to set it.
    fetch('/sendVisualizationData')
        .then(function (response) {
            return response.text();
        }).then(function (text) {
        console.log('GET response text:');
        console.log(text); // Print the greeting as text.
    });

    // Send the same request.
    fetch('/sendVisualizationData')
        .then(function (response) {
            return response.json(); // But parse it as JSON this time.
        })
        .then(function (json) {
            console.log('GET response as JSON:');
            console.log(json); // Here’s our JSON object.
        });
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Function to draw a simple word tree.
 *
 * Word Tree Code URL: https://bl.ocks.org/d3noob/1a96af738c89b88723eb63456beb6510
 *
 * TODO: figure out how to construct the necessary data structure/format when using actual live data.
 */
visualization_concept.drawWordTree = () => {
    let treeData =
        {
            "name": "Hello",
            "children": [
                {
                    "name": "-World, which may seem unprofessional on the surface (or",
                    "children": [
                        {"name": " at least to me); I hope we can discuss",
                            "children": [
                                {"name": " it later if something interesting comes up"},
                                {"name": " more on the differences between the"}
                            ]
                        },
                        {"name": " not) but will get our project published quicker as"}
                    ]
                },
                {"name": "-world package is actually part of some existing project."},
                {"name": "-everyone, thanks for watching the second part of my new story;"}
            ]
        };


    // Clear svg before drawing another.
    d3.selectAll("svg > *").remove();

    // Set the dimensions and margins of the diagram
    let margin = {top: 20, right: 90, bottom: 30, left: 90},
        width = 1024 - margin.left - margin.right,
        height = 768 - margin.top - margin.bottom;

    // append the svg object to the body of the page (removed append)
    // appends a 'group' element to 'svg'
    // moves the 'group' element to the top left margin
    let svg = d3.select("#visualization-svg")
        .attr("width", width + margin.right + margin.left)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate("
            + margin.left + "," + margin.top + ")");

    //Set dimensions of canvas to same dimensions as svg.
    d3.select("#visualization-canvas")
        .attr("width", width + margin.right + margin.left)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate("
            + margin.left + "," + margin.top + ")");

    // Ghetto way to redraw background image on canvas after resizing in d3.
    visualization_concept.context.clearRect(0, 0, width + margin.right + margin.left, height + margin.top + margin.bottom);
    visualization_concept.context.drawImage(background_image, 0, 0,
        width + margin.right + margin.left,height + margin.top + margin.bottom);

    let i = 0,
        duration = 750,
        root;

    // declares a tree layout and assigns the size
    let treemap = d3.tree().size([height, width]);

    // Assigns parent, children, height, depth
    root = d3.hierarchy(treeData, function (d) {
        return d.children;
    });
    root.x0 = height / 2;
    root.y0 = 0;

    // Collapse after the second level
    root.children.forEach(collapse);

    update(root);

    // Collapse the node and all it's children
    function collapse(d) {
        if (d.children) {
            d._children = d.children;
            d._children.forEach(collapse);
            d.children = null
        }
    }

    function update(source) {

        // Assigns the x and y position for the nodes
        // noinspection JSValidateTypes
        let treeData = treemap(root);

        // Compute the new tree layout.
        let nodes = treeData.descendants(),
            links = treeData.descendants().slice(1);

        // Normalize for fixed-depth.
        nodes.forEach(function (d) {
            d.y = d.depth * 180
        });

        // ****************** Nodes section ***************************

        // Update the nodes...
        let node = svg.selectAll('g.node')
            .data(nodes, function (d) {
                return d.id || (d.id = ++i);
            });

        // Enter any new modes at the parent's previous position.
        let nodeEnter = node.enter().append('g')
            .attr('class', 'node')
            .attr("transform", function () {
                return "translate(" + source.y0 + "," + source.x0 + ")";
            })
            .on('click', click);

        // Add Circle for the nodes
        nodeEnter.append('circle')
            .attr('class', 'node')
            .attr('r', 1e-6)
            .style("fill", function (d) {
                return d._children ? "lightsteelblue" : "#fff";
            });

        // Add labels for the nodes
        nodeEnter.append('text')
            .attr("dy", ".35em")
            .attr("x", function (d) {
                return d.children || d._children ? -13 : 13;
            })
            .attr("text-anchor", function (d) {
                return d.children || d._children ? "end" : "start";
            })
            .text(function (d) {
                return d.data.name;
            });

        // UPDATE
        let nodeUpdate = nodeEnter.merge(node);

        // Transition to the proper position for the node
        nodeUpdate.transition()
            .duration(duration)
            .attr("transform", function (d) {
                return "translate(" + d.y + "," + d.x + ")";
            });

        // Update the node attributes and style
        nodeUpdate.select('circle.node')
            .attr('r', 10)
            .style("fill", function (d) {
                return d._children ? "lightsteelblue" : "#c0adff";
            })
            .attr('cursor', 'pointer');


        // Remove any exiting nodes
        let nodeExit = node.exit().transition()
            .duration(duration)
            .attr("transform", function () {
                return "translate(" + source.y + "," + source.x + ")";
            })
            .remove();

        // On exit reduce the node circles size to 0
        nodeExit.select('circle')
            .attr('r', 1e-6);

        // On exit reduce the opacity of text labels
        nodeExit.select('text')
            .style('fill-opacity', 1e-6);

        // ****************** links section ***************************

        // Update the links...
        let link = svg.selectAll('path.link')
            .data(links, function (d) {
                return d.id;
            });

        // Enter any new links at the parent's previous position.
        let linkEnter = link.enter().insert('path', "g")
            .attr("class", "link")
            .attr('d', function () {
                let o = {x: source.x0, y: source.y0};
                return diagonal(o, o)
            });

        // UPDATE
        let linkUpdate = linkEnter.merge(link);

        // Transition back to the parent element position
        linkUpdate.transition()
            .duration(duration)
            .attr('d', function (d) {
                return diagonal(d, d.parent)
            });

        // Remove any exiting links
        link.exit().transition()
            .duration(duration)
            .attr('d', function () {
                let o = {x: source.x, y: source.y};
                return diagonal(o, o)
            })
            .remove();
    // Store the old positions for transition.
        nodes.forEach(function (d) {
            d.x0 = d.x;
            d.y0 = d.y;
        });

        // Creates a curved (diagonal) path from parent to the child nodes
        function diagonal(s, d) {

            return `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`
        }

        // Toggle children on click.
        function click(d) {
            if (d.children) {
                d._children = d.children;
                d.children = null;
            } else {
                d.children = d._children;
                d._children = null;
            }
            update(d);
        }
    }
};