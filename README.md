# NYC Multimodal Ride-Hailing Data Retrieval Platform

## Project Introduction
New York City sees approximately 23 million taxi ride records each month, it equivalent to over 700,000 ride requests daily. For taxi companies, the ability to accurately dispatch vehicles directly impacts their operational performance and market share. This project aims to analyze the New York City Taxi & Limousine Commission (TLC) Trip Record Data to develop a big data platform equipped with predictive and retrieval functions for ride demand. 

## Project Presentation and Video
- Presentation: [Presentation](https://github.com/TicyYang/NYC_Multimodal_Ride-Hailing_Data_Retrieval_Platform/blob/main/presentation.md)
- Video:
  - [Full Video](https://www.youtube.com/watch?v=-n5lJlV2XCI)
  - [Web and Visualization Dashboard Features](https://youtu.be/8SKvn0fmuLI)

## Directory & File Description
- data_preprocessing: Scripts for data preprocessing and feature engineering.
- data_timeseries: Scripts and data for converting time series data, with TS6 version used for machine learning training and TS10 version used for prediction.
- datasets: Datasets used for feature engineering and visualization.
- demo: Demo video of the website, also accessible via the link in the **Project Presentation and Video** section to watch on YouTube.
- machine_learning: Scripts for machine learning training and prediction.
- presentation: Project presentation file, viewable directly by clicking on `presentation.md`.
- spark_performance_testing: Scripts for Spark performance testing.
- web: Frontend and backend code for the demo website.
- web_server_installation: Configuration files and procedures for setting up the web server on GCP.


## Tools Used (Versions)
<table>
  <thead>
    <tr>
      <th>Usage</th>
      <th>Tool</th>
      <th>Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="text-align: center;"><strong>Data Collection</strong></td>
      <td>Requests</td>
      <td>2.31.0</td>
    </tr>
    <tr>
      <td>BeautifulSoup4</td>
      <td>4.12.2</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center;"><strong>Data Preprocessing</strong></td>
      <td>Numpy</td>
      <td>1.25.2</td>
    </tr>
    <tr>
      <td>Pandas</td>
      <td>1.5.3</td>
    </tr>
    <tr>
      <td>PyArrow</td>
      <td>11.0.0</td>
    </tr>
    <tr>
      <td rowspan="4" style="text-align: center;"><strong>Environment Setup</strong></td>
      <td>Ubuntu</td>
      <td>22.04 LTS</td>
    </tr>
    <tr>
      <td>VMware Workstation Player</td>
      <td>17.0</td>
    </tr>
    <tr>
      <td>Hadoop</td>
      <td>3.3.6</td>
    </tr>
    <tr>
      <td>Spark</td>
      <td>3.2.4</td>
    </tr>
    <tr>
      <td rowspan="2" style="text-align: center;"><strong>Machine Learning Model Building</strong></td>
      <td>scikit-learn</td>
      <td>1.2.2</td>
    </tr>
    <tr>
      <td>PySpark</td>
      <td>3.4.1</td>
    <tr>
      <td rowspan="4" style="text-align: center;"><strong>Visualization</strong></td>
      <td>Flask</td>
      <td>2.3.2</td>
    </tr>
    <tr>
      <td>Leaflet</td>
      <td>1.9.4</td>
    </tr>
    <tr>
      <td>Plotly</td>
      <td>5.16.1</td>
    </tr>
    <tr>
      <td>Dash</td>
      <td>2.12.1</td>
    </tr>
  </tbody>
</table>

