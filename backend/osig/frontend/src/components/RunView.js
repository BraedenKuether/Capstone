import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage } from 'formik';
import { ResponsiveLine } from '@nivo/line';
import { ResponsiveBar } from '@nivo/bar'
import {PriceEarnings,ModelReturns,
        CumulativeReturns,DividendYield,
        PriceShares, TopBottom} from "./Graphs.js"

class RunView extends Component {
  constructor(props) {
    super(props);
    this.state = {
      graph_returns: [],
      graph_cumreturns: [],
      loaded: false,
      placeholder: "Loading",
      data: null,
      weights_str: "",
      currentGraph:"CR"
    };
    this.getGraphChange = this.getGraph.bind(this);
  }
  
  componentDidMount() {
    let url = window.location.toString().split('/')
    let id = url[url.length - 2]
    var reqInit = {method: 'GET'};
    let api_endpoint = window.location.origin.concat("/api/portfolio_analysis/get_runs/".concat(id));
    var req = new Request(api_endpoint, reqInit);
    fetch(req)
      .then(response => response.json())
      .then(data => {
        console.log(data);
        let weights_str = "";
        for(let i = 0; i < data.tickers.length; i++) {
          weights_str += data.tickers[i] + ": " + data.pred[0].weights[i] + "\n"
        }
        this.setState({
          graph_returns: data.pred,
          graph_cumreturns: data.cumreturns,
          data: data,
          loaded: true,
          weights_str: weights_str
        });
        console.log(this.state.data.alphabeta[0]);
    })
  }
  
  getGraph(event){
    this.setState({currentGraph : event.target.value});
  } 

  render() {
    if (this.state.data === null) {
      return null;
    }
    let graph;
    switch (this.state.currentGraph) {
      
      case "CR":
        graph = <CumulativeReturns state={this.state}/>;
        break;

      case "MR":
        graph = <ModelReturns state={this.state}/>;
        break; 

      case "TB":
        graph = <TopBottom state={this.state}/>;
        break;

      case "PE":
        graph = <PriceEarnings state={this.state}/>;
        break;

      case "PS":
        graph = <PriceShares state={this.state}/>;
        break;

      case "DY":
        graph = <DividendYield state={this.state}/>;
        console.log("fell thru");
        break;
    }
    return (
      <div>
        <br/>
        <br/>
        <select className="form-select" onChange={this.getGraphChange}>
          <option value="CR">Cumulative Returns</option>
          <option value="MR">Model Returns</option>
          <option value="TB">Top Bottom Performers</option>
          <option value="PE">Price Earnings</option>
          <option value="PS">Price Shares</option>
          <option value="DY">Dividend Yield</option>
        </select>
        <div>
          {graph}
        </div>
        <div class="row">
            <h2 class="text-center">Portfolio & Weights</h2>
            <div>
              {this.state.weights_str}
            </div>
        </div>
        
        <br/>
        <br/>

        <table className="table table-sm">
          <thead>
            <th scope="col">Sharpe Ratio</th>
            <th scope="col">Risk & Variance</th>
            <th scope="col">Alpha & Beta</th>
          </thead>
          <tbody>
            <tr>
              <td> {this.state.data.sharperatio} </td>
              <td> {this.state.data.portrisk[0]} {this.state.data.portrisk[1]} </td>
              <td> {this.state.data.alphabeta[0]} {this.state.data.alphabeta[1]} </td>
            </tr>
          </tbody>
        </table>
        
        <table className="table table-sm">
          <thead>
            <th scope="col">Total Performance</th>
            <th scope="col">YTD Performance</th>
            <th scope="col">S&P Performance</th>
          </thead>
          <tbody>
            <tr>
              <td> {this.state.data.totalperf}</td>
              <td> {this.state.data.ytdperf} </td>
              <td> {this.state.data.spytd} </td>
            </tr>
          </tbody>
        </table>
      </div>);
  }
}

export default RunView;
