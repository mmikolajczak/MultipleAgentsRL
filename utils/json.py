import json


def load_json_file(json_path):
    with open(json_path) as f:
        config = json.loads(f.read())
    return config


def save_json_file(json_path, data_dict):
    with open(json_path, 'w') as f:
        f.write(json.dumps(data_dict))
