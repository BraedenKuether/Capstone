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
      .then(response => {
        if(response.status != 200) {
          this.setState({
            errorMsg: "Database Error: Could Not Fetch Runs",
            loaded: false
          });
        }
        else {
          response.json()
          .then(data => {
            this.setState({
              runs: data.data,
              loaded: true
            })
          });
        }
      });
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
           if (values.ticker == []) {
             errors.tickers = 'Must Have at Least One Ticker';
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
                    <table>
                    <tr>
                      <th>Ticker Name</th>
                      <th>Weighting in Portfolio (Optional)</th> 
                      <th>Remove Ticker</th>
                      <th>Add ticker</th>
                    </tr>
                    <tbody>
                    {props.values.tickers.map((ticker, index) => {
                      return(
                      <tr key = {index}>
                        <td>
                        <span style={{display:"inline-block", width: "87px"}}></span>
                        <Field 
                        type = "text"
                        name={`tickers.${index}.name`} />
                        </td>
                        <td>
                        <Field 
                        type = "float"
                        name={`tickers.${index}.weight`} />
                        </td>
                        <td>
                        <br/>
                        <button
                         type="button"
                         onClick={() => arrayHelpers.remove(index)} // remove a friend from the list
                        >
                        -
                        </button>
                        </td>
                        <td>
                        <button
                         type="button"
                         onClick={() => arrayHelpers.insert(index, '')} // insert an empty string at a position
                        >
                        +
                        </button>
                        </td>
                     </tr>
                    );
                    })}
                   </tbody>
                   </table>
                 ) : (
                   <button type="button" onClick={() => arrayHelpers.push('')}>
                     {/* show this when user has removed all friends from the list */}
                     Add a Ticker
                   </button>
                 )}
                </div>
             )}
             />
              <br/>
              <label>Title &emsp;</label>
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
            <br/>
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
