import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage, FieldArray } from 'formik';
import {ResponsiveLine} from '@nivo/line';

const TableHeader = (props) => {
    const {headers} = props;
    return(
      <thead className="table-active" key="header-1">
        <tr key="header-0">
          { headers && headers.map((value, index) => {
              return <th key={index}><div>{value.charAt(0).toUpperCase() + value.slice(1)}</div></th>
          })}
        </tr>
      </thead>
    );
  }

const  TableBody = (props) => {
    const { headers, rows } = props;

    function buildRow(row, headers) {
        return (
             <tr key={row.id}>
             { headers.map((value, index) => {
                 if (index == 0) {
                  let url = 'view_run/'.concat(row.id);
                  return <td key={index}><a href={url}>{row[value]}</a></td>
                 } else {
                  return <td key={index}>{row[value]}</td>
                 } 
              })}
             </tr>
         )
      };

      return(
          <tbody>
            { rows && rows.map((value) => {
                    return buildRow(value, headers);
                })}
          </tbody>
    );
  }

const  Table = (props) => {
    const { headers, rows } = props;
    return (
      <div>
      <table className="table table-bordered table-hover">
      <TableHeader headers={headers}></TableHeader>
      <TableBody headers={headers} rows={rows}></TableBody>
      </table>
      </div>
    );
  }

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
  
  

  renderTable() {
    if(this.state.loaded && this.state.runs.length > 0) {
      return (
      <div className="container p-2">
        <div className="row">
          <div className="col">
            <Table headers={Object.keys(this.state.runs[0])} rows={this.state.runs} />
          </div>
        </div>
      </div>
      );
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
              errorMsg: "One or more of the tickers could not be found",
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
                    props.values.tickers.map((ticker, index) => (
                      <div key={index}>
                        <div>
                        <label>Ticker Name</label>
                        <span style={{display:"inline-block", width: "87px"}}></span>
                        <Field 
                        type = "text"
                        name={`tickers.${index}.name`} />
                        </div>
                        <br/>
                        <div>
                        <label>Weighting in Portfolio &emsp;</label>
                        <Field 
                        type = "float"
                        name={`tickers.${index}.weight`} />
                        </div>
                        <br/>
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
                        <br/>
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
        <br/>
        <br/>
        {this.state.loaded &&
          <div>
          <h1 id='title' className="text-center">Previous Runs</h1>
          {this.renderTable()}
          </div>
        }
      </div>
    );
  }
}


export default PortfolioAnalysis;
