# Execution Statements
This project is denotes the full process of ML Process for Moniroting & Scoring and reporting metrics.

## Configuration
 1. This project will use configuration JSON to read the configurations that are required for store the data , models and metrics and reporting metrics as well.
 2. For dependencies resolution we will use requirements.txt file.
 ```
 pip install -r requirements.txt

 ```

 ## Running individual Scripts:
 ```
    python ingestion.py
    python training.py
    python scoring.py
    python deployment.py 
    python diagnostics.py 
    python reporting.py 

```
    ingestion.py is required to ingesting the data.
    training.py is required to train the model and store the model in pickle file
    scoring.py  is required to calculate the scoring metrics like F1 Score.
    deployment.py  is useful to deploy the model into production
    diagnostics.py  is giving full report for opertaional issues.
    reporting.py  call the repoting endpoints to present the meircs data.

## Testing App Endpoints uisng CommandLine Interface
```
   curl localhost:8000/
   curl localhost:8000/prediction
   curl localhost:8000/scoring
   curl localhost:8000/summarystats
   curl localhost:8000/diagnostics
```

## API calls using python request models

```
python apicalls.py
```

## Automated Full Process
```
python fullprocess.py

```

Schdeuling crontab using cron
To automate the process we will use cron, we have to open start the cron service and edit the cron.
```
crontab -e
```

Add the below schedule line cron 
```
*/10 * * * * /usr/bin/env python /path/to/fullprocess.py 

```