import falcon
from tasks import getSug
from json import dumps
from celery.result import AsyncResult
from falcon_cors import CORS

cors_allow_all = CORS(allow_all_origins=True, allow_all_headers=True, allow_credentials_all_origins=True,
                      allow_all_methods=True)

app = falcon.API(middleware=[cors_allow_all.middleware])


class TaskStatus(object):
    cors = cors_allow_all

    def on_get(self, req, resp, task_id):
        # get result of task by task_id and generate content to client
        task_result = AsyncResult(task_id)
        if task_result.status == 'SUCCESS':
            resp.status = falcon.HTTP_CREATED
            resp.content_type = 'image/jpeg'
            resp.body = task_result.result
        else:
            resp.status = falcon.HTTP_200
            resp.body = dumps({'status':task_result.status,'task_id':task_id})


class StartTask(object):
    cors = cors_allow_all

    def on_post(self, req, resp):
        svg = req.stream.read().decode("utf-8")
        if len(svg) == 0:
            resp.status = falcon.HTTP_422
        else:
            task = getSug.delay(svg)
            resp.status = falcon.HTTP_ACCEPTED
            result = {'task_id': task.id}
            resp.body = dumps(result)


app.add_route('/start_task', StartTask())
app.add_route('/task_status/{task_id}', TaskStatus())
