# Similar Logos
This is a basic API to be used with [logobot](https://github.com/mezaugusto/logobot)

## Prerequisites
You need a model and the dataset in order to run the scheduler.

## Configure
Run `configure.py --help` to see the options available

## Run
1. Run the server
`gunicorn server:app`

2. Run `logobot` in the branch `suggestions`

3. Run the celery task scheduler
`celery -A tasks worker --loglevel=info`

## Endpoints

There are two endpoints available.

`start_task/` expects form data with an svg image in text format, returns task_id


`/task_status/{task_id}` return 201 if task is finished, 200 otherwise