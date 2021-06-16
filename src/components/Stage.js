import React from "react";

class Stage extends React.Component {
    render() {
        const { stage } = this.props;
        return (
            <h3>
                {stage}
            </h3>
        );
    }
}

export default Stage;
