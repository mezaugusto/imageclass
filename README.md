# imageclass
Image classification for the logobot similar logos recommendations

1. Run the server

`gunicorn server:app`

2. Run the Logobot in the branch suggestions

3. Run the celery task scheduler

`celery -A tasks worker --loglevel=info`