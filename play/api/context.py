import contextvars


# Context variables for current request
request_id_ctx = contextvars.ContextVar('request_id', default=None)