import React, { Component } from "react";
import { render } from "react-dom";
import { Formik, Form, Field, ErrorMessage } from 'formik';

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
    console.log(values);
    var reqInit = {method: 'POST', body: values};
    console.log(reqInit)
    var req = new Request("api/portfolio_analysis", reqInit);
    console.log(req);
    fetch(req).then(function(response) {
      return response.blob();
    }).then(function(response) {
      console.log('fetched');
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
              <Field type="checkbox" name="checked" value="Predictions" />
              Predictions
              </label>
              <label>
              <Field type="checkbox" name="checked" value="Alpha" />
              Alpha
              </label>
              <label>
              <Field type="checkbox" name="checked" value="Beta" />
              Beta
              </label>
            </div>
            <button type="submit">Submit</button>
          </Form>
        )
        }
        </Formik>
      </div>
    );
  }
}


export default PortfolioAnalysis;