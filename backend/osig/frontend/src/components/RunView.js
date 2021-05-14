import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage } from 'formik';
import { ResponsiveLine } from '@nivo/line';
import { ResponsiveBar } from '@nivo/bar'

class RunView extends Component {
  constructor(props) {
    super(props);
    this.state = {
      graph_returns: [],
      graph_cumreturns: [],
      loaded: false,
      placeholder: "Loading",
      data: null,
      weights_str: ""
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

  render() {
    if (this.state.data === null) {
      return null;
    }
    return (
      
      <div className='ViewContainer'>
      <h2>Recommended Weighting</h2>
      <div>
      {this.state.weights_str}
      </div>
      <h2>Simulated Returns With ML Model</h2>
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
        <h2>Cumalitive Returns With Inputted Weights</h2>
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
        <h2>Dividend Yield</h2>
        <div style={{height:'600px',width:'1200px'}}>
        <ResponsiveBar
            data={this.state.data.dividendyield}
            margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
            keys={['returns']}
            indexBy="ticker"
            height = {500}
            padding={0.3}
            valueScale={{ type: 'linear' }}
            indexScale={{ type: 'band', round: true }}
            colors={{ scheme: 'nivo' }}
            defs={[
                {
                    id: 'dots',
                    type: 'patternDots',
                    background: 'inherit',
                    color: '#38bcb2',
                    size: 4,
                    padding: 1,
                    stagger: true
                },
                {
                    id: 'lines',
                    type: 'patternLines',
                    background: 'inherit',
                    color: '#eed312',
                    rotation: -45,
                    lineWidth: 6,
                    spacing: 10
                }
            ]}
            borderColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            axisTop={null}
            axisRight={null}
            axisBottom={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'ticker',
                legendPosition: 'middle',
                legendOffset: 32
            }}
            axisLeft={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'return',
                legendPosition: 'middle',
                legendOffset: -40
            }}
            labelSkipWidth={12}
            labelSkipHeight={12}
            labelTextColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            legends={[
                {
                    dataFrom: 'keys',
                    anchor: 'bottom-right',
                    direction: 'column',
                    justify: false,
                    translateX: 120,
                    translateY: 0,
                    itemsSpacing: 2,
                    itemWidth: 100,
                    itemHeight: 20,
                    itemDirection: 'left-to-right',
                    itemOpacity: 0.85,
                    symbolSize: 20,
                    effects: [
                        {
                            on: 'hover',
                            style: {
                                itemOpacity: 1
                            }
                        }
                    ]
                }
            ]}
            animate={true}
            motionStiffness={90}
            motionDamping={15}
        />
        </div>
        <h2>Price Per Earnings</h2>
        <div style={{height:'600px',width:'1200px'}}>
        <ResponsiveBar
            data={this.state.data.priceearnings[0]}
            margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
            keys={['returns']}
            indexBy="ticker"
            height = {500}
            padding={0.3}
            valueScale={{ type: 'linear' }}
            indexScale={{ type: 'band', round: true }}
            colors={{ scheme: 'nivo' }}
            defs={[
                {
                    id: 'dots',
                    type: 'patternDots',
                    background: 'inherit',
                    color: '#38bcb2',
                    size: 4,
                    padding: 1,
                    stagger: true
                },
                {
                    id: 'lines',
                    type: 'patternLines',
                    background: 'inherit',
                    color: '#eed312',
                    rotation: -45,
                    lineWidth: 6,
                    spacing: 10
                }
            ]}
            borderColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            axisTop={null}
            axisRight={null}
            axisBottom={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'ticker',
                legendPosition: 'middle',
                legendOffset: 32
            }}
            axisLeft={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'return',
                legendPosition: 'middle',
                legendOffset: -40
            }}
            labelSkipWidth={12}
            labelSkipHeight={12}
            labelTextColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            legends={[
                {
                    dataFrom: 'keys',
                    anchor: 'bottom-right',
                    direction: 'column',
                    justify: false,
                    translateX: 120,
                    translateY: 0,
                    itemsSpacing: 2,
                    itemWidth: 100,
                    itemHeight: 20,
                    itemDirection: 'left-to-right',
                    itemOpacity: 0.85,
                    symbolSize: 20,
                    effects: [
                        {
                            on: 'hover',
                            style: {
                                itemOpacity: 1
                            }
                        }
                    ]
                }
            ]}
            animate={true}
            motionStiffness={90}
            motionDamping={15}
        />
        </div>
        <h2>Price to Sales Ratios</h2>
        <div style={{height:'600px',width:'1200px'}}>
        <ResponsiveBar
            data={this.state.data.priceshares[0]}
            margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
            keys={['returns']}
            indexBy="ticker"
            height = {500}
            padding={0.3}
            valueScale={{ type: 'linear' }}
            indexScale={{ type: 'band', round: true }}
            colors={{ scheme: 'nivo' }}
            defs={[
                {
                    id: 'dots',
                    type: 'patternDots',
                    background: 'inherit',
                    color: '#38bcb2',
                    size: 4,
                    padding: 1,
                    stagger: true
                },
                {
                    id: 'lines',
                    type: 'patternLines',
                    background: 'inherit',
                    color: '#eed312',
                    rotation: -45,
                    lineWidth: 6,
                    spacing: 10
                }
            ]}
            borderColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            axisTop={null}
            axisRight={null}
            axisBottom={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'ticker',
                legendPosition: 'middle',
                legendOffset: 32
            }}
            axisLeft={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'return',
                legendPosition: 'middle',
                legendOffset: -40
            }}
            labelSkipWidth={12}
            labelSkipHeight={12}
            labelTextColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            legends={[
                {
                    dataFrom: 'keys',
                    anchor: 'bottom-right',
                    direction: 'column',
                    justify: false,
                    translateX: 120,
                    translateY: 0,
                    itemsSpacing: 2,
                    itemWidth: 100,
                    itemHeight: 20,
                    itemDirection: 'left-to-right',
                    itemOpacity: 0.85,
                    symbolSize: 20,
                    effects: [
                        {
                            on: 'hover',
                            style: {
                                itemOpacity: 1
                            }
                        }
                    ]
                }
            ]}
            animate={true}
            motionStiffness={90}
            motionDamping={15}
        />
        </div>
        <h2>Performance Per Share</h2>
        <div style={{height:'600px',width:'1200px'}}>
        <ResponsiveBar
            data={this.state.data.topbottomperf}
            margin={{ top: 50, right: 130, bottom: 50, left: 60 }}
            keys={['returns']}
            indexBy="ticker"
            height = {500}
            padding={0.3}
            valueScale={{ type: 'linear' }}
            indexScale={{ type: 'band', round: true }}
            colors={{ scheme: 'nivo' }}
            defs={[
                {
                    id: 'dots',
                    type: 'patternDots',
                    background: 'inherit',
                    color: '#38bcb2',
                    size: 4,
                    padding: 1,
                    stagger: true
                },
                {
                    id: 'lines',
                    type: 'patternLines',
                    background: 'inherit',
                    color: '#eed312',
                    rotation: -45,
                    lineWidth: 6,
                    spacing: 10
                }
            ]}
            borderColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            axisTop={null}
            axisRight={null}
            axisBottom={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'ticker',
                legendPosition: 'middle',
                legendOffset: 32
            }}
            axisLeft={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'return',
                legendPosition: 'middle',
                legendOffset: -40
            }}
            labelSkipWidth={12}
            labelSkipHeight={12}
            labelTextColor={{ from: 'color', modifiers: [ [ 'darker', 1.6 ] ] }}
            legends={[
                {
                    dataFrom: 'keys',
                    anchor: 'bottom-right',
                    direction: 'column',
                    justify: false,
                    translateX: 120,
                    translateY: 0,
                    itemsSpacing: 2,
                    itemWidth: 100,
                    itemHeight: 20,
                    itemDirection: 'left-to-right',
                    itemOpacity: 0.85,
                    symbolSize: 20,
                    effects: [
                        {
                            on: 'hover',
                            style: {
                                itemOpacity: 1
                            }
                        }
                    ]
                }
            ]}
            animate={true}
            motionStiffness={90}
            motionDamping={15}
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
        <h2>Risk (Variance/STD)</h2>
        <div>
        {this.state.data.portrisk[0]}/{this.state.data.portrisk[1]}
        </div>
        <div>
        <h2>YTD Performance</h2>
        <div>
        {this.state.data.ytdperf}
        </div>
        <h2>Alpha/Beta</h2>
        <div>
        {this.state.data.alphabeta[0]}/{this.state.data.alphabeta[1]}
        </div>
        </div>
        
      </div>
    );
  }
}

export default RunView;