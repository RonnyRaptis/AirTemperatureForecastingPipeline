The purpose of this program is to predict the average teamperature of each month for the next 12 months based on climate data from Jan 1st 2000 to Dec 24 2024 from the ESA Capernicus satellite
If you want to use this program yourself you will have to go the  website below (you will have to make an account) and select 2m Air Temperature, the years 2000-2024, every month, every day, the time 12:00, and set the geographical area to whole available region. 
Download it as a zip file, extract the file and rename it "era5_t2m.nc" (this file may be upwards of 15 gigabytes). Add a folder to the project labeled "datasets" and place the .nc file in that folder, don't forget to install dependencies and use python 3.13 64bit

https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
