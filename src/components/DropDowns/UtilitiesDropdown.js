import React from "react";

import SelectSearch from 'react-select-search';
import { utilities } from '../data'
import fuzzySearch from "../../utilities/fuzzySearch";

class UtilitiesDropDown extends React.Component {
    render() {
        return (
            <div>
                <p><u>Utilities:</u></p>
                <SelectSearch
                    options={utilities}
                    search
                    filterOptions={fuzzySearch}
                    onChange={(value) => this.props.pickUtility(value)}
                />
            </div>
        );
    }
}

export default UtilitiesDropDown;