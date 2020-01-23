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
    // Bind functions to event handlers.
    $('#generate-button').bind('click', visualization.generate);

    // Setup canvas for output.
    visualization.canvas = $('#visualization-canvas')[0];
    visualization.context = visualization.canvas.getContext('2d');
    visualization.context.fillStyle = 'rgb(255,0,0)';

    // Load images.
    visualization.loadImages();
    // Draw images.
    visualization.drawBackground();

    // Draw the word tree.
    // visualization.drawWordTree();
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

/**
 * Generate visualization button.
 */
visualization.generate = () => {
    // Test import data as JSON.
    visualization.importDatasetAsJson();
    // Test import CSV using D3.
    visualization.importDatasetWithD3();
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
visualization.importDatasetWithD3 = () => {
    // noinspection SpellCheckingInspection
    d3.csv("/static/files/next_token_logits_test.csv").then(function (data) {
        processData(data);
    }).catch(function (error) {
        console.log(error);
        console.log('Failed to import data.');
    });
    d3.selectAll('svg#visualization');
};

/**
 * Function to import the data as JSON from Flask web server using GET.
 */
visualization.importDatasetAsJson = () => {
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
 */
visualization.drawWordTree = () => {
    var m = [20, 120, 20, 120],
        w = 1280 - m[1] - m[3],
        h = 800 - m[0] - m[2],
        i = 0,
        root;

    var tree = d3.layout.tree()
        .size([h, w]);

    var diagonal = d3.svg.diagonal()
        .projection(function(d) { return [d.y, d.x]; });

    var vis = d3.select("#body").append("svg:svg")
        .attr("width", w + m[1] + m[3])
        .attr("height", h + m[0] + m[2])
        .append("svg:g")
        .attr("transform", "translate(" + m[3] + "," + m[0] + ")");

    d3.json("flare.json", function(json) {
        root = json;
        root.x0 = h / 2;
        root.y0 = 0;

        function toggleAll(d) {
            if (d.children) {
                d.children.forEach(toggleAll);
                toggle(d);
            }
        }

        // Initialize the display to show a few nodes.
        root.children.forEach(toggleAll);
        update(root);
    });

    function update(source) {
        var duration = d3.event && d3.event.altKey ? 5000 : 500;

        // Compute the new tree layout.
        var nodes = tree.nodes(root).reverse();

        // Normalize for fixed-depth.
        nodes.forEach(function(d) { d.y = d.depth * 180; });

        // Update the nodes…
        var node = vis.selectAll("g.node")
            .data(nodes, function(d) { return d.id || (d.id = ++i); });

        // Enter any new nodes at the parent's previous position.
        var nodeEnter = node.enter().append("svg:g")
            .attr("class", "node")
            .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
            .on("click", function(d) { toggle(d); update(d); });

        nodeEnter.append("svg:circle")
            .attr("r", 1e-6)
            .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

        nodeEnter.append("svg:text")
            .attr("x", function(d) { return d.children || d._children ? -10 : 10; })
            .attr("dy", ".35em")
            .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
            .text(function(d) { return d.name; })
            .style("fill-opacity", 1e-6);

        // Transition nodes to their new position.
        var nodeUpdate = node.transition()
            .duration(duration)
            .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

        nodeUpdate.select("circle")
            .attr("r", 4.5)
            .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

        nodeUpdate.select("text")
            .style("fill-opacity", 1);

        // Transition exiting nodes to the parent's new position.
        var nodeExit = node.exit().transition()
            .duration(duration)
            .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
            .remove();

        nodeExit.select("circle")
            .attr("r", 1e-6);

        nodeExit.select("text")
            .style("fill-opacity", 1e-6);

        // Update the links…
        var link = vis.selectAll("path.link")
            .data(tree.links(nodes), function(d) { return d.target.id; });

        // Enter any new links at the parent's previous position.
        link.enter().insert("svg:path", "g")
            .attr("class", "link")
            .attr("d", function(d) {
                var o = {x: source.x0, y: source.y0};
                return diagonal({source: o, target: o});
            })
            .transition()
            .duration(duration)
            .attr("d", diagonal);

        // Transition links to their new position.
        link.transition()
            .duration(duration)
            .attr("d", diagonal);

        // Transition exiting nodes to the parent's new position.
        link.exit().transition()
            .duration(duration)
            .attr("d", function(d) {
                var o = {x: source.x, y: source.y};
                return diagonal({source: o, target: o});
            })
            .remove();

        // Stash the old positions for transition.
        nodes.forEach(function(d) {
            d.x0 = d.x;
            d.y0 = d.y;
        });
    }

// Toggle children.
    function toggle(d) {
        if (d.children) {
            d._children = d.children;
            d.children = null;
        } else {
            d.children = d._children;
            d._children = null;
        }
    }
};