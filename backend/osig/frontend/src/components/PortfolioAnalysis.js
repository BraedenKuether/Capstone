import React, { Component } from "react";
import { render } from "react-dom";
import {ResponsiveLine} from '@nivo/line';
import data from "./data";
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";

class PortfolioAnalysis extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      loaded: false,
      placeholder: "Loading"
    };
  }

  componentDidMount() {
    fetch("api/lead")
      .then(response => {
        if (response.status > 400) {
          return this.setState(() => {
            return { placeholder: "Something went wrong!" };
          });
        }
        return response.json();
      })
      .then(data => {
        this.setState(() => {
          return {
            data,
            loaded: true
          };
        });
      });
  }

  render() {
    return (
      <div style={{height:'600px',width:'1200px'}}>
      <h1>Portfolio Analysis</h1>
      </div>
    );
  }
}


export default PortfolioAnalysis;