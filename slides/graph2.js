
var $ = go.GraphObject.make;
var myDiagram =
  $(go.Diagram, "diagram",
    {
      initialContentAlignment: go.Spot.Center, // center Diagram contents
      "undoManager.isEnabled": true // enable Ctrl-Z to undo and Ctrl-Y to redo
    });

var myModel = $(go.Model);
// in the model data, each node is represented by a JavaScript object:
myModel.nodeDataArray = [
  { key: "Alpha" },
  { key: "Beta" },
  { key: "Gamma" }
];
myDiagram.model = myModel;
    