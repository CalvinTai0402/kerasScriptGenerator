import React from "react";

import SelectSearch from 'react-select-search';
import { optimizers } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class OptimizerDropDown extends React.Component {
    render() {
        return (
            <div>
                <p><u>Optimizers:</u></p>
                <SelectSearch
                    options={optimizers}
                    search
                    filterOptions={fuzzySearch}
                    onChange={(value) => this.props.handleChange("OPTIMIZERS", value)}
                    placeholder="Select an optimizer" />
            </div>
        );
    }
}

export default OptimizerDropDown;