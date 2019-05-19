import json

def extract_data(request):
  json_string = request.body.decode()
  return json.loads(json_string)
