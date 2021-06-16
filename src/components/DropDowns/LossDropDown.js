import React from "react";
import SelectSearch from 'react-select-search';

import { losses } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class LossDropDown extends React.Component {
    render() {
        return (
            <div>
                <p><u>Losses:</u></p>
                <SelectSearch
                    search
                    filterOptions={fuzzySearch}
                    options={losses}
                    onChange={(value) => this.props.handleChange("LOSS", value)}
                    placeholder="Select a loss" />
            </div>
        );
    }
}

export default LossDropDown;