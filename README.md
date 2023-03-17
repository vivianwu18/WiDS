# Women in Data Science (WiDS Datathon) 2023
## Background
Extreme weather events are sweeping the globe and range from heat waves, wildfires and drought to hurricanes, extreme rainfall and flooding. These weather events have multiple impacts on agriculture, energy, transportation, as well as low resource communities and disaster planning in countries across the globe.

Accurate long-term forecasts of temperature and precipitation are crucial to help people prepare and adapt to these extreme weather events. Currently, purely physics-based models dominate short-term weather forecasting. But these models have a limited forecast horizon. The availability of meteorological data offers an opportunity for data scientists to improve sub-seasonal forecasts by blending physics-based forecasts with machine learning. Sub-seasonal forecasts for weather and climate conditions (lead-times ranging from 15 to more than 45 days) would help communities and industries adapt to the challenges brought on by climate change.

## Overview
This year’s datathon, organized by the WiDS Worldwide team at Stanford University, Harvard University IACS, Arthur, and the WiDS Datathon Committee, will focus on longer-term weather forecasting to help communities adapt to extreme weather events caused by climate change.

The dataset was created in collaboration with Climate Change AI (CCAI). Participants will submit forecasts of temperature and precipitation for one year, competing against the other teams as well as official forecasts from NOAA.

## Evaluation Metric
The evaluation metric for this competition is Root Mean Squared Error (RMSE). The RMSE is a commonly used measure of the differences between predicted values provided by a model and the actual observed values.

RMSE is computed as:

$$RMSE = \sqrt{{1\over{N}} {\sum_{n=1}^N} (y^n - \hat{y}^{(n)})^2}$$

where $y^{(n)}$ is the n-th observed value and $\hat{y}^{(n)}$ is the n-th predicted value given by the model.

## Model Results

| Model Name | Final RMSE |
| ---------- | ---------- |
| CATboost | 1.14 |
| XGBOOST | 1.61 | 
| Linear Regression | 6.31 |
| PCA transform | 13.76 | 
