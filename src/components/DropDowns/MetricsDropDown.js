import React from "react";
import SelectSearch from 'react-select-search';

import { metrics } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class MetricsDropDown extends React.Component {
    state = {
        count: 1
    }

    render() {
        const { count } = this.state;
        return (
            <div>
                <p><u>Metrics:</u></p>
                {Array(count).fill(<SelectSearch
                    search
                    filterOptions={fuzzySearch}
                    options={metrics}
                    onChange={(value) => this.props.handleChange("METRICS", value)}
                    placeholder="Select a metric"
                    printOptions="on-focus" />)}
                <button className="button" onClick={() => this.setState({ count: count + 1 })}>
                    Add another metric
                </button>
            </div>
        );
    }
}

export default MetricsDropDown;