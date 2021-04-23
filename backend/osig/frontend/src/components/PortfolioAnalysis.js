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
    var reqInit = {
      method: 'GET',
    };
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
    if(this.state.loaded && this.state.runs.length > 0) {
      let header = Object.keys(this.state.runs[0])
      return header.map((key, index) => {
        return <th key={index}>{key.toUpperCase()}</th>
      })
    } else {
      return <th></th>
    }
  }
  
  renderTableData() {
    if(this.state.loaded && this.state.runs.length > 0) {
      return this.state.runs.map((run, index) => {
        const { id, title, date } = run //destructuring
        let id_url = 'view_run/'.concat(id)
        return (
          <tr key={id}>
            <td><a href={id_url}>{title}</a></td>
            <td>{date}</td>
            <td>{id}</td>
          </tr>
         )
      })
    } else {
      return (
        <tr>
          <td></td>
        </tr>
      )
    }
  }
  
  getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
  }
  
  submit(values) {
    let csrftoken = this.getCookie('csrftoken');
    var reqInit = {
      method: 'POST', 
      body: JSON.stringify(values),
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFTOKEN': csrftoken
      },
    };
    console.log(values);
    let api_endpoint = window.location.origin.concat("/api/portfolio_analysis/create_run");
    console.log("submitting ".concat(api_endpoint));
    var req = new Request(api_endpoint, reqInit);
    fetch(req)
      .then(response => response.text())
      .then(data => {
        this.componentDidMount();
    })
  }

  render() {
    return (
      <div className='PortfolioContainer'>
        <h1>Portfolio Analysis</h1>
        <Formik
         initialValues={{ 
          tickers: '', 
          title: ''
         }}
         validate={values => {
           const errors = {};
           if (!values.tickers) {
             errors.tickers = 'Required';
           }
           if (!values.title) {
             errors.title = 'Required';
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
              <label>Title</label>
              <Field
               type="text"
               name="title"
               id="title"
               placeholder="Name This Run"
               onChange={props.handleChange}
               onBlur={props.handleBlur}
               value={props.values.title}
               />
               <ErrorMessage
                component="div"
                name="title"
                className="invalid-feedback"
              />
            </div>
            <button type="submit">Submit</button>
          </Form>
        )
        }
        </Formik>
        {this.state.loaded &&
          <div>
          <h1 id='title'>Previous Runs</h1>
          <table id='runs'>
            <tbody>
              <tr>{this.renderTableHeader()}</tr>
              {this.renderTableData()}
            </tbody>
          </table>
          </div>
        }
      </div>
    );
  }
}


export default PortfolioAnalysis;
