import falcon
from tasks import getSug
from json import dumps
from celery.result import AsyncResult


class StartTask(object):
    def on_post(self, req, resp):
        # start task
        task = getSug.delay(req.get_param('data'))
        resp.status = falcon.HTTP_ACCEPTED
        # return task_id to client
        result = {'task_id': task.id}
        resp.body = dumps(result)


class TaskStatus(object):
    def on_get(self, req, resp, task_id):
        # get result of task by task_id and generate content to client
        task_result = AsyncResult(task_id)
        if task_result.status == 'SUCCESS':
            resp.status = falcon.HTTP_CREATED
            resp.content_type = 'image/jpeg'
            resp.body = task_result.result
        else:
            resp.status = falcon.HTTP_200
            resp.body = dumps(task_result.status)


app = falcon.API()
app.req_options.auto_parse_form_urlencoded = True

app.add_route('/start_task', StartTask())
app.add_route('/task_status/{task_id}', TaskStatus())