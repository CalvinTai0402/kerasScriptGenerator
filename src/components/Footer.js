import React from "react";
/* eslint-disable jsx-a11y/accessible-emoji */

export default function Footer() {
    return (
        <div className="footer-div">
            <p className="footer-text" style={{ color: "black" }}>
                ----- Made by <a href="https://calvintai0402.github.io/index.html#/">Calvin</a> -----
            </p>
            <p className="footer-text" style={{ color: "black" }}>
                This is in development. Most of the Keras code belongs to François Chollet. All PRs are welcome @ <a href="https://github.com/CalvinTai0402/kerasScriptGenerator">repo</a>
            </p>
        </div>
    );
}