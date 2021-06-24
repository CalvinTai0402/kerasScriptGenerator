import React from "react";
import SelectSearch from 'react-select-search';

import { metrics } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class MetricsDropDown extends React.Component {
    state = {
        metricsList: [[<SelectSearch search filterOptions={fuzzySearch} options={metrics} onChange={(value) => this.handleChange(value, 0)} placeholder="Select a layer" printOptions="on-focus" />, ""]]
    }

    handleChange = (value, index) => {
        const { metricsList } = this.state;
        metricsList[index][1] = value;
        this.setState({ metricsList }, () => {
            let metricsToSet = [];
            for (let i = 0; i < metricsList.length; i++) {
                metricsToSet.push(metricsList[i][1])
            }
            this.props.handleChange("METRICS", metricsToSet);
        })
    }

    handleButtonClicked = () => {
        const { metricsList } = this.state;
        let nextLayerIndex = metricsList.length
        metricsList.push([<SelectSearch search filterOptions={fuzzySearch} options={metrics} onChange={(value) => this.handleChange(value, nextLayerIndex)} placeholder="Select a layer" printOptions="on-focus" />, ""])
        this.setState({ metricsList })

    }

    render() {
        const { metricsList } = this.state;
        let displayedMetrics = metricsList.map((layer) =>
            layer[0]
        );
        return (
            <div>
                <p><u>Metrics:</u></p>
                {displayedMetrics}
                <button className="button" onClick={this.handleButtonClicked}>
                    Add another layer
                </button>
            </div>
        );
    }
}

export default MetricsDropDown;