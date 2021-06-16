import React from "react";
import SelectSearch from 'react-select-search';

import { layers } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class LayersDropDown extends React.Component {
    state = {
        layersList: [[<SelectSearch search filterOptions={fuzzySearch} options={layers} onChange={(value) => this.handleChange(value, 0)} placeholder="Select a layer" printOptions="on-focus" />, ""]]
    }

    handleChange = (value, index) => {
        const { layersList } = this.state;
        layersList[index][1] = value;
        this.setState({ layersList }, () => {
            let layersToSet = [];
            for (let i = 0; i < layersList.length; i++) {
                layersToSet.push(layersList[i][1])
            }
            this.props.handleChange("LAYERS", layersToSet);
        })
    }

    handleButtonClicked = () => {
        const { layersList } = this.state;
        let nextLayerIndex = layersList.length
        layersList.push([<SelectSearch search filterOptions={fuzzySearch} options={layers} onChange={(value) => this.handleChange(value, nextLayerIndex)} placeholder="Select a layer" printOptions="on-focus" />, ""])
        this.setState({ layersList })

    }

    render() {
        const { layersList } = this.state;
        let displayedLayers = layersList.map((layer) =>
            layer[0]
        );
        return (
            <div>
                <p><u>Layers:</u></p>
                {displayedLayers}
                <button className="button" onClick={this.handleButtonClicked}>
                    Add another layer
                </button>
            </div>
        );
    }
}

export default LayersDropDown;