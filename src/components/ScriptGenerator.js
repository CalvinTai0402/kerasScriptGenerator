import React from "react";
import LayersDropDown from "./DropDowns/LayersDropDown";
import OptimizerDropDown from "./DropDowns/OptimizerDropDown";
import LossDropDown from "./DropDowns/LossDropDown";
import MetricsDropDown from "./DropDowns/MetricsDropDown";
import CallbacksDropDown from "./DropDowns/CallbacksDropDown";
import ExampleDropDown from "./DropDowns/ExampleDropDown"
import DataPreprocessingDropDown from "./DropDowns/DataPreprocessingDropDown"
import Stage from "./Stage"
import { fileContent, imageDataPreprocessing, vectorizeSequence } from './data'
import { MNISTCategoricalClassification, MNISTCategoricalClassificationWithCNN, IMDBBinaryClassification, BostonHousingRegression, MNISTCategoricalClassificationWithTransferLearningAndFineTuning } from "./example"

class ScriptGenerator extends React.Component {
  state = {
    fileContent: fileContent,
    example: MNISTCategoricalClassification,
    placeholderValuePairs: {
      "DATAPREPROCESSING": "",
      "OPTIMIZERS": "",
      "LOSS": "",
      "METRICS": [],
      "LAYERS": [],
      "CALLBACKS": [],
    }
  };

  handleChange = (placeholder, value) => {
    const { placeholderValuePairs } = this.state;
    placeholderValuePairs[placeholder] = value;
    this.setState({ placeholderValuePairs })
  }

  updateFileContent = async () => {
    const { placeholderValuePairs, fileContent } = this.state;
    let fileContentTemp = fileContent;
    for (let [placeholder, value] of Object.entries(placeholderValuePairs)) {
      if ((placeholder === "METRICS" && placeholderValuePairs["METRICS"].length > 1) || (placeholder === "CALLBACKS" && placeholderValuePairs["CALLBACKS"].length > 1) || (placeholder === "LAYERS" && placeholderValuePairs["LAYERS"].length > 1)) {
        value = value.join(",\n")
      }
      if (placeholder !== "DATAPREPROCESSING") {
        fileContentTemp = fileContentTemp.replace(placeholder, value);
      }
    }
    switch (this.state.placeholderValuePairs["DATAPREPROCESSING"]) {
      case "imageDataPreprocessing":
        console.log(imageDataPreprocessing)
        fileContentTemp = fileContentTemp.replace("DATAPREPROCESSING", imageDataPreprocessing);
        break;
      case "vectorizeSequence":
        fileContentTemp = fileContentTemp.replace("DATAPREPROCESSING", vectorizeSequence);
        break;
      default:
        break;
    }
    this.setState({ fileContent: fileContentTemp })
  }

  reset = () => {
    window.location.reload();
  }

  downloadScript = async () => {
    await this.updateFileContent();
    const element = document.createElement("a");
    const file = new Blob([this.state.fileContent], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "script.py";
    document.body.appendChild(element); // Required for this to work in FireFox
    element.click();
    this.reset();
  }

  pickExample = (exampleTitle) => {
    switch (exampleTitle) {
      case "IMDBBinaryClassification":
        this.setState({ example: IMDBBinaryClassification })
        break;
      case "MNISTCategoricalClassification":
        this.setState({ example: MNISTCategoricalClassification })
        break;
      case "MNISTCategoricalClassificationWithCNN":
        this.setState({ example: MNISTCategoricalClassificationWithCNN })
        break;
      case "BostonHousingRegression":
        this.setState({ example: BostonHousingRegression })
        break;
      case "MNISTCategoricalClassificationWithTransferLearningAndFineTuning":
        this.setState({ example: MNISTCategoricalClassificationWithTransferLearningAndFineTuning })
        break;
      default:
        this.setState({ example: MNISTCategoricalClassification })
    }
  }

  downloadExample = async () => {
    const element = document.createElement("a");
    const file = new Blob([this.state.example], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "example.py";
    document.body.appendChild(element); // Required for this to work in FireFox
    element.click();
  }

  render() {
    return (
      <React.Fragment>
        <Stage stage={"Data preprocessing:"} />
        <DataPreprocessingDropDown handleChange={this.handleChange} />
        <Stage stage={"Model Building:"} />
        <CallbacksDropDown handleChange={this.handleChange} />
        <LayersDropDown handleChange={this.handleChange} />
        <Stage stage={"Model Compiling:"} />
        <OptimizerDropDown handleChange={this.handleChange} />
        <LossDropDown handleChange={this.handleChange} />
        <MetricsDropDown handleChange={this.handleChange} />
        <div>
          <button className="button" onClick={this.downloadScript}>Download Script</button>
        </div>
        <Stage stage={"Download examples:"} />
        <ExampleDropDown pickExample={this.pickExample} />
        <div>
          <button className="button" onClick={this.downloadExample}>Download Example</button>
        </div>

      </React.Fragment>

    );
  }
}

export default ScriptGenerator;