import React from "react";
import SelectSearch from 'react-select-search';

import { callbacks } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class CallbacksDropDown extends React.Component {
    state = {
        callbacksList: [[<SelectSearch search filterOptions={fuzzySearch} options={callbacks} onChange={(value) => this.handleChange(value, 0)} placeholder="Select a callback" printOptions="on-focus" />, ""]]
    }

    handleChange = (value, index) => {
        const { callbacksList } = this.state;
        callbacksList[index][1] = value;
        this.setState({ callbacksList }, () => {
            let callbacksToSet = [];
            for (let i = 0; i < callbacksList.length; i++) {
                callbacksToSet.push(callbacksList[i][1])
            }
            this.props.handleChange("CALLBACKS", callbacksToSet);
        })
    }

    handleButtonClicked = () => {
        const { callbacksList } = this.state;
        let nextCallbackIndex = callbacksList.length
        callbacksList.push([<SelectSearch search filterOptions={fuzzySearch} options={callbacks} onChange={(value) => this.handleChange(value, nextCallbackIndex)} placeholder="Select a callback" printOptions="on-focus" />, ""])
        this.setState({ callbacksList })

    }

    render() {
        const { callbacksList } = this.state;
        let displayedCallbacks = callbacksList.map((callback) =>
            callback[0]
        );
        return (
            <div>
                <p><u>Callbacks:</u></p>
                {displayedCallbacks}
                <button className="button" onClick={this.handleButtonClicked}>
                    Add another callback
                </button>
            </div>
        );
    }
}

export default CallbacksDropDown;