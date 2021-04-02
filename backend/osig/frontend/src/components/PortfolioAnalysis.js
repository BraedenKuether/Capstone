import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage } from 'formik';
import {ResponsiveLine} from '@nivo/line';

class PortfolioAnalysis extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      loaded: false,
      placeholder: "Loading"
    };
  }
  
  submit(values) {
    var reqInit = {method: 'POST', body: JSON.stringify(values)};
    var req = new Request("api/portfolio_analysis", reqInit);
    fetch(req)
      .then(response => response.json())
      .then(data => {
        console.log(data.pred);
        this.setState({
          data: data.pred
        });
    })
  }
  
  componentDidMount() {
    this.setState(() => {
      return {
        loaded: true
      };
    });
  }

  render() {
    return (
      <div className='temp'>
        <h1>Portfolio Analysis</h1>
        <Formik
         initialValues={{ 
          tickers: '', 
          checked: []
         }}
         validate={values => {
           const errors = {};
           if (!values.tickers) {
             errors.tickers = 'Required';
           }
           return errors;
         }}
        onSubmit={(values, { setSubmitting }) => {
                 this.submit(values)
                 setSubmitting(false);
         }}>
        {props => (
          <Form onSubmit={props.handleSubmit}>
            <div>
              <label style={{ display: "block" }}>Tickers</label>
              <Field
               type="text"
               name="tickers"
               id="tickers"
               placeholder="Enter your tickers separated by commas"
               onChange={props.handleChange}
               onBlur={props.handleBlur}
               value={props.values.tickers}
               />
              <ErrorMessage
                component="div"
                name="tickers"
                className="invalid-feedback"
              />
              <label>
              <Field type="checkbox" name="checked" value="pred" />
              Predictions
              </label>
              <label>
              <Field type="checkbox" name="checked" value="alphabeta" />
              Alpha Beta
              </label>
            </div>
            <button type="submit">Submit</button>
          </Form>
        )
        }
        </Formik>
      <div style={{height:'600px',width:'1200px'}}>
       <ResponsiveLine
        data={this.state.data}
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
                    legend: 'transportation',
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
      </div>
    );
  }
}


export default PortfolioAnalysis;
