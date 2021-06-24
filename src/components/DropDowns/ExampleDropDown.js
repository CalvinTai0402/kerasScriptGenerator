import React from "react";

import SelectSearch from 'react-select-search';
import { examples } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class OptimizerDropDown extends React.Component {
    render() {
        return (
            <div>
                <p><u>Examples:</u></p>
                <SelectSearch
                    options={examples}
                    search
                    filterOptions={fuzzySearch}
                    onChange={(value) => this.props.pickExample(value)}
                />
            </div>
        );
    }
}

export default OptimizerDropDown;