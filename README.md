# RAS Matrix Balancer

This is a Streamlit web app that allows users to balance a matrix using the RAS (Row and Column Scaling) method. Upload your Excel matrix and margin data, run the balancing, and download the result.

## Features
- Upload matrix and margin data from Excel files
- Perform RAS balancing with custom tolerance
- View error logs and convergence status
- Download the balanced matrix

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy

## Run Locally
```bash
pip install -r requirements.txt
streamlit run ras_app.py
