import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage } from 'formik';
import { ResponsiveLine } from '@nivo/line';
import { ResponsiveBar } from '@nivo/bar'


export function PriceEarnings(props){
    if (props.state.data.priceearnings != null) {
      return (<div data-value="PE">
            <div style={{height:'600px',width:'1200px'}}>
        <ResponsiveBar
            data={props.state.data.priceearnings[0]}
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
        </div>);
        
    }
    return null;
}

export function ModelReturns(props) { 
      
      return(
        <div data-value="MR">
        <div style={{height:'600px',width:'1200px'}}>
         <ResponsiveLine
          data={props.state.graph_returns}
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
        </div>);
}

const isMonth = value => (Number(value.split("-")[3]) % 20) == 0;
const getDate = value => (value.split("-")[0] + "-" + value.split("-")[1] + "-" + value.split("-")[2])
export function CumulativeReturns(props) {
  var arr = props.state.graph_cumreturns;
  var i = 0;
  for (i = 0; i < arr[0]['data'].length; i++) {
    arr[0]["data"][i]["x"] = arr[0]["data"][i]["x"]+"-"+i.toString();
  }
  return (<div data-value="CR">
         <div style={{height:'600px',width:'1200px'}}>
         <ResponsiveLine
          data={arr}
           margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
            xScale={{ type: 'point'}}
             yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: true, reverse: false }}
              yFormat=" >-.2f"
               axisTop={null}
                axisRight={null}
                 axisBottom={{
                    format: function(value) {
                      return isMonth(value) ? getDate(value) : "";
                    },
                    tickSize: function(value) {
                      return isMonth(value) ? 5 : 0;
                    },
                    orient: 'bottom',
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
                               legend: 'Returns',
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
</div>);
}

export function DividendYield(props) {
  if (props.state.dividendyield != null) {
      return (<div data-value="DY">
        <div style={{height:'600px',width:'1200px'}}>
        <ResponsiveBar
            data={props.state.data.dividendyield}
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
        </div>);
  }
  return null;
}

export function PriceShares(props) {
  
  if (props.state.data.priceshares != null) {
    return (<div data-value="PS">
        <div style={{height:'600px',width:'1200px'}}>
        <ResponsiveBar
            data={props.state.data.priceshares[0]}
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
    </div>);

  }
  return null;
} 

export function TopBottom(props) {
  return (<div data-value="TB">
            <div style={{height:'600px',width:'1200px'}}>
            <ResponsiveBar
            data={props.state.data.topbottomperf}
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
          </div>);
}
