import React, { Component } from "react";
import ReactDOM from 'react-dom';
import { render } from "react-dom";
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";
import Graph from "./Graph";
import PortfolioAnalysis from "./PortfolioAnalysis";

class App extends Component {
    render() {
    return (
    <Router>
        <Switch>
            <Route exact path="/" component={Graph} />
            <Route exact path="/portfolio_analysis" component={PortfolioAnalysis} />
        </Switch>
    </Router>
    );
  }
}

export default App;