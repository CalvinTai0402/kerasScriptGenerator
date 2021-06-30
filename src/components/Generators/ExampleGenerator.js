import React from "react";
import ExampleDropDown from "../DropDowns/ExampleDropDown"
import Stage from "../Stage"
import {
    MNISTCategoricalClassification, MNISTCategoricalClassificationWithCNN, IMDBBinaryClassification,
    BostonHousingRegression, MNISTCategoricalClassificationWithTransferLearningAndFineTuning, OxfordPetsImageSegmentation
} from "../example"

class ExampleGenerator extends React.Component {
    state = {
        example: MNISTCategoricalClassification
    };

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
            case "OxfordPetsImageSegmentation":
                this.setState({ example: OxfordPetsImageSegmentation })
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
                <Stage stage={"Download examples:"} />
                <ExampleDropDown pickExample={this.pickExample} />
                <div>
                    <button className="button" onClick={this.downloadExample}>Download Example</button>
                </div>
            </React.Fragment>
        );
    }
}

export default ExampleGenerator;