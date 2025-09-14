from werkzeug.wrappers import Response

def handler(request):
    return Response("KLT Backend API is running.", mimetype="text/plain")
