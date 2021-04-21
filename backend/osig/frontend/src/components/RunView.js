import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage } from 'formik';
import {ResponsiveLine} from '@nivo/line';

class RunView extends Component {
  constructor(props) {
    super(props);
    this.state = {
      graph_returns: [],
      graph_cumreturns: [],
      loaded: false,
      placeholder: "Loading",
      data: {}
    };
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
        this.setState({
          graph_returns: data.pred,
          graph_cumreturns: data.cumreturns,
          data: data,
          loaded: true,
        });
    })
  }

  render() {
    return (
      <div className='ViewContainer'>
        <div style={{height:'600px',width:'1200px'}}>
         <ResponsiveLine
          data={this.state.graph_returns}
           margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
            xScale={{ type: 'point' }}
             yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: true, reverse: false }}
              yFormat=" >-.2f"
               axisTop={null}
                axisRight={null}
                 axisBottom={{
                  orient: 'bottom',
                   tickSize: 5,
                    tickPadding: 5,
                     tickRotation: 0,
                      legend: 'Date',
                       legendOffset: 36,
                        legendPosition: 'middle'
                         }}
                          axisLeft={{
                           orient: 'left',
                            tickSize: 5,
                             tickPadding: 5,
                              tickRotation: 0,
                               legend: 'count',
                                legendOffset: -40,
                                 legendPosition: 'middle'
                                  }}
                                   pointSize={10}
                                    pointColor={{ theme: 'background' }}
                                     pointBorderWidth={2}
                                      pointBorderColor={{ from: 'serieColor' }}
                                       pointLabelYOffset={-12}
                                        useMesh={true}
                                         legends={[
                                          {
                                           anchor: 'bottom-right',
                                            direction: 'column',
                                             justify: false,
                                              translateX: 100,
                                               translateY: 0,
                                                itemsSpacing: 0,
                                                 itemDirection: 'left-to-right',
                                                  itemWidth: 80,
                                                   itemHeight: 20,
                                                    itemOpacity: 0.75,
                                                     symbolSize: 12,
                                                      symbolShape: 'circle',
                                                       symbolBorderColor: 'rgba(0, 0, 0, .5)',
                                                        effects: [
                                                         {
                                                          on: 'hover',
                                                           style: {
                                                            itemBackground: 'rgba(0, 0, 0, .03)',
                                                             itemOpacity: 1
                                                              }
                                                               }
                                                                ]
                                                                 }
                                                                  ]}
                                                                   />
        </div>
         <div style={{height:'600px',width:'1200px'}}>
         <ResponsiveLine
          data={this.state.graph_cumreturns}
           margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
            xScale={{ type: 'point' }}
             yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: true, reverse: false }}
              yFormat=" >-.2f"
               axisTop={null}
                axisRight={null}
                 axisBottom={{
                  orient: 'bottom',
                   tickSize: 5,
                    tickPadding: 5,
                     tickRotation: 0,
                      legend: 'Date',
                       legendOffset: 36,
                        legendPosition: 'middle'
                         }}
                          axisLeft={{
                           orient: 'left',
                            tickSize: 5,
                             tickPadding: 5,
                              tickRotation: 0,
                               legend: 'count',
                                legendOffset: -40,
                                 legendPosition: 'middle'
                                  }}
                                   pointSize={10}
                                    pointColor={{ theme: 'background' }}
                                     pointBorderWidth={2}
                                      pointBorderColor={{ from: 'serieColor' }}
                                       pointLabelYOffset={-12}
                                        useMesh={true}
                                         legends={[
                                          {
                                           anchor: 'bottom-right',
                                            direction: 'column',
                                             justify: false,
                                              translateX: 100,
                                               translateY: 0,
                                                itemsSpacing: 0,
                                                 itemDirection: 'left-to-right',
                                                  itemWidth: 80,
                                                   itemHeight: 20,
                                                    itemOpacity: 0.75,
                                                     symbolSize: 12,
                                                      symbolShape: 'circle',
                                                       symbolBorderColor: 'rgba(0, 0, 0, .5)',
                                                        effects: [
                                                         {
                                                          on: 'hover',
                                                           style: {
                                                            itemBackground: 'rgba(0, 0, 0, .03)',
                                                             itemOpacity: 1
                                                              }
                                                               }
                                                                ]
                                                                 }
                                                                  ]}
                                                                   />
        </div>
        <div>
        <h2>Sharpe Ratio</h2>
        <div>
        {this.state.data.sharperatio}
        </div>
        </div>
        <div>
        <h2>SPYTD</h2>
        <div>
        {this.state.data.spytd}
        </div>
        </div>
        <div>
        <h2>Total Performance</h2>
        <div>
        {this.state.data.totalperf}
        </div>
        </div>
        <div>
        <h2>YTD Performance</h2>
        <div>
        {this.state.data.ytdperf}
        </div>
        </div>
      </div>
    );
  }
}

export default RunView;