import React from "react";
import SelectSearch from 'react-select-search';

import { dataPreprocessing } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class DataPreprocessingDropDown extends React.Component {
    render() {
        return (
            <div>
                <p><u>Data preprocessing:</u></p>
                <SelectSearch
                    search
                    filterOptions={fuzzySearch}
                    options={dataPreprocessing}
                    onChange={(value) => this.props.handleChange("DATAPREPROCESSING", value)}
                    placeholder="Select a preprocessing method" />
            </div>
        );
    }
}

export default DataPreprocessingDropDown;