# config directory

Directory which contains configuration files for Grafana, Prometheus and cAdvisor,
which are used for scraping metrics from containers as part of benchmarking.

## File overview

### dashboard_index.json

This file is used to shape the Grafana dashboard which is displayed. 
I have attempted to display F1-score in addition to what was originally displayed.

### dashboards.yml

This file is to change various aspects of the Grafana dashboard unrelated to a specific dashboard,
e.g., change where to look for the dashboard file, how often it should be updated and so on.

### prometheus-datasources.yml

This file is used to make Grafana use data from Prometheus in the dashboard.

### grafana.ini

This is simply metadata for Grafana, 
pointing to where the dashboard_index.json file is located
and skipping the user authentication process.

### prometheus.yml

This file configures where and how Prometheus scrapes data.
Currently the file tells Prometheus to scrape data from cAdvisor and from the server.