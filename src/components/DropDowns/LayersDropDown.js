import React from "react";
import SelectSearch from 'react-select-search';

import { layers } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class MetricsDropDown extends React.Component {
    state = {
        count: 1
    }

    render() {
        const { count } = this.state;
        return (
            <div>
                <p><u>Layers:</u></p>
                {Array(count).fill(<SelectSearch
                    search
                    filterOptions={fuzzySearch}
                    options={layers}
                    onChange={(value) => this.props.handleChange("LAYERS", value)}
                    placeholder="Select a layer"
                    printOptions="on-focus" />)}
                <button className="button" onClick={() => this.setState({ count: count + 1 })}>
                    Add another layer
                </button>
            </div>
        );
    }
}

export default MetricsDropDown;