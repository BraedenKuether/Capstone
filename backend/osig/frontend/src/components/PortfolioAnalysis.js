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
      placeholder: "Loading",
      runs: []
    };
  }

  componentDidMount() {
    var reqInit = {method: 'GET'};
    var req = new Request(window.location.origin.concat("/api/portfolio_analysis/get_runs/all"), reqInit);
    fetch(req)
      .then(response => response.json())
      .then(data => {
        this.setState({
          runs: data.data,
          loaded: true
        });
    })
  }
  
  renderTableHeader() {
    if(this.state.loaded) {
      console.log(this.state.runs);
      let header = Object.keys(this.state.runs[0])
      return header.map((key, index) => {
        return <th key={index}>{key.toUpperCase()}</th>
      })
    } else {
      return <th></th>
    }
  }
  
  renderTableData() {
    return this.state.runs.map((run, index) => {
      const { id, title, date } = run //destructuring
      let id_url = 'view_run/'.concat(id)
      return (
        <tr key={id}>
          <td>{id}</td>
          <td><a href={id_url}>{title}</a></td>
          <td>{date}</td>
        </tr>
       )
    })
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

  render() {
    return (
      <div className='PortfolioContainer'>
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
        <h1 id='title'>Previous Runs</h1>
        <table id='students'>
          <tbody>
            <tr>{this.renderTableHeader()}</tr>
            {this.renderTableData()}
          </tbody>
        </table>
      </div>
    );
  }
}


export default PortfolioAnalysis;
