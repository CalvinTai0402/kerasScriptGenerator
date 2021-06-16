import React from "react";
import LayersDropDown from "./DropDowns/LayersDropDown";
import OptimizerDropDown from "./DropDowns/OptimizerDropDown";
import LossDropDown from "./DropDowns/LossDropDown";
import MetricsDropDown from "./DropDowns/MetricsDropDown";
import Stage from "./Stage"
import { fileContent } from './data'


class ScriptGenerator extends React.Component {
  state = {
    fileContent: fileContent,
    placeholderValuePairs: {
      "OPTIMIZERS": "",
      "LOSS": "",
      "METRICS": [],
      "LAYERS": []
    }
  };

  handleChange = (placeholder, value) => {
    const { placeholderValuePairs } = this.state;
    if (placeholder === "METRICS" && !(placeholderValuePairs["METRICS"].includes(value))) {
      placeholderValuePairs[placeholder].push(value);
    } else if (placeholder !== "METRICS") {
      placeholderValuePairs[placeholder] = value;
    }
    this.setState({ placeholderValuePairs })
  }

  updateFileContent = async () => {
    const { placeholderValuePairs, fileContent } = this.state;
    let fileContentTemp = fileContent;
    for (let [placeholder, value] of Object.entries(placeholderValuePairs)) {
      if (placeholder === "METRICS" && placeholderValuePairs["METRICS"].length > 1 || placeholder === "LAYERS" && placeholderValuePairs["LAYERS"].length > 1) {
        value = value.join(", ")
      }
      fileContentTemp = fileContentTemp.replace(placeholder, value);
    }
    this.setState({ fileContent: fileContentTemp })
  }

  reset = () => {
    window.location.reload();
  }

  downloadTxtFile = async () => {
    await this.updateFileContent();
    const element = document.createElement("a");
    const file = new Blob([this.state.fileContent], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "script.py";
    document.body.appendChild(element); // Required for this to work in FireFox
    element.click();
    this.reset();
  }

  render() {
    return (
      <React.Fragment>
        <Stage stage={"Model Building:"} />
        <LayersDropDown handleChange={this.handleChange} />
        <Stage stage={"Model Compiling:"} />
        <OptimizerDropDown handleChange={this.handleChange} />
        <LossDropDown handleChange={this.handleChange} />
        <MetricsDropDown handleChange={this.handleChange} />
        <div>
          <button className="button" onClick={this.downloadTxtFile}>Download Script</button>
        </div>
      </React.Fragment>

    );
  }
}

export default ScriptGenerator;