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
import RunView from "./RunView";

class App extends Component {
    render() {
    return (
    <Router>
        <Switch>
            <Route exact path="/" component={Graph} />
            <Route exact path="/portfolio_analysis" component={PortfolioAnalysis} />
            <Route path="/portfolio_analysis/view_run" component={RunView} />
        </Switch>
    </Router>
    );
  }
}

export default App;