from os import sep
from helpers.NumpyEncoder import NumpyEncoder
import json

def write_json(cur_data, output_path, json_filename):
    # Save as json file
    with open(output_path + sep + json_filename, 'w') as cur_json:
        cur_json.write(json.dumps(cur_data, cls=NumpyEncoder, indent=4))