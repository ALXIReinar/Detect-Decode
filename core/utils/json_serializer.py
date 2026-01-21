from datetime import datetime, date
import json
from decimal import Decimal
from uuid import UUID


def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert(item) for item in obj)
    elif isinstance(obj, set):
        return list(convert(item) for item in obj)
    elif isinstance(obj, (Decimal, UUID, datetime, date)):
        return str(obj)
    return obj

def json_dump(obj):
    return json.dumps(convert(obj)).encode('utf-8')

def json_loads(obj):
    return json.loads(obj)
