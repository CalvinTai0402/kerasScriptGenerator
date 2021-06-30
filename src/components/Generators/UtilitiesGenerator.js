import React from "react";
import Stage from "../Stage"
import UtilitiesDropDown from "../DropDowns/UtilitiesDropdown"
import {
    colabFileUploadDownload
} from "../utilities"

class ExampleGenerator extends React.Component {
    state = {
        utility: colabFileUploadDownload
    };

    pickUtility = (utilityTitle) => {
        switch (utilityTitle) {
            case "colabFileUploadDownload":
                this.setState({ utility: colabFileUploadDownload })
                break;
            default:
                this.setState({ utility: colabFileUploadDownload })
        }
    }

    downloadUtility = async () => {
        const element = document.createElement("a");
        const file = new Blob([this.state.utility], { type: 'text/plain' });
        element.href = URL.createObjectURL(file);
        element.download = "example.py";
        document.body.appendChild(element); // Required for this to work in FireFox
        element.click();
    }

    render() {
        return (
            <React.Fragment>
                <Stage stage={"Download Utilities:"} />
                <UtilitiesDropDown pickUtility={this.pickUtility} />
                <div>
                    <button className="button" onClick={this.downloadUtility}>Download Utility</button>
                </div>
            </React.Fragment>
        );
    }
}

export default ExampleGenerator;