import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage, FieldArray } from 'formik';
import {ResponsiveLine} from '@nivo/line';

class PortfolioAnalysis extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      loaded: false,
      placeholder: "Loading",
      runs: [],
      errorMsg: '',
      successMsg: '',
      submitting: ''
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
    this.setState({
      errorMsg: '',
      successMsg: '',
      submitting: 'Submitting...'
    });
    let csrftoken = this.getCookie('csrftoken');
    var reqInit = {
      method: 'POST', 
      body: JSON.stringify(values),
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFTOKEN': csrftoken
      },
    };
    let api_endpoint = window.location.origin.concat("/api/portfolio_analysis/create_run");
    console.log("submitting ".concat(api_endpoint));
    var req = new Request(api_endpoint, reqInit);
    fetch(req)
      .then(response => {
        if(response.status != 200) {
          response.text().then(data => {
            this.setState({
              errorMsg: data,
              successMsg: '',
              submitting: ''
            });
            this.componentDidMount();
          })
        } else {
          response.text().then(data => {
            this.setState({
              errorMsg: '',
              successMsg: data,
              submitting: ''
            });
            this.componentDidMount();
          })
        }
      })
  }

  render() {
    return (
      <div className='PortfolioContainer'>
        <h1>Portfolio Analysis</h1>
        <Formik
         initialValues={{ 
          tickers: [],
          title: '',
          numTickers:0,
         }}
         validate={values => {
           const errors = {};
           
           
           if (!values.title) {
             errors.title = 'Required';
           }
           console.log(errors)
           return errors;
         }}
        onSubmit={(values, { setSubmitting }) => {
                 this.submit(values)
                 setSubmitting(false);
         }}>
         {(props, setValues ) => (
          <Form onSubmit={props.handleSubmit}>
            <div>
            <FieldArray
             name="tickers"
             render={arrayHelpers => (
                <div>
                  {props.values.tickers && props.values.tickers.length > 0 ? (
                    props.values.tickers.map((ticker, index) => (
                      <div key={index}>
                        <div>
                        <label>Ticker Name</label>
                        <Field 
                        type = "text"
                        name={`tickers.${index}.name`} />
                        </div>
                        <div>
                        <label>Weighting in Portfolio</label>
                        <Field 
                        type = "float"
                        name={`tickers.${index}.weight`} />
                        </div>
                        <button
                         type="button"
                         onClick={() => arrayHelpers.remove(index)} // remove a friend from the list
                        >
                        -
                        </button>
                        <button
                         type="button"
                         onClick={() => arrayHelpers.insert(index, '')} // insert an empty string at a position
                        >
                        +
                        </button>
                     </div>
                   ))
                 ) : (
                   <button type="button" onClick={() => arrayHelpers.push('')}>
                     {/* show this when user has removed all friends from the list */}
                     Add a Ticker
                   </button>
                 )}
                </div>
             )}
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
        <div style={{color: 'red'}}>
          {this.state.errorMsg}
        </div>
        <div style={{color: 'green'}}>
          {this.state.successMsg}
        </div>
        <div>
          {this.state.submitting}
        </div>
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
