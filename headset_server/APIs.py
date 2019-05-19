from emokit.headset import Headset
from headset_server.helper_functions import *
from django.http import HttpResponse
import json

headset = Headset()

def is_ready(request):
    global headset
    sample = headset.get_sample()
    if sample == None:
        headset = Headset()
        response = {
        "ready": "false",
        "sample": sample
        }
    else:
        response = {
        "ready": "true",
        "sample": sample
        }

    return HttpResponse(json.dumps(response))
