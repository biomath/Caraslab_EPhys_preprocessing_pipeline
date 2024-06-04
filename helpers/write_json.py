from os import sep
from helpers.NumpyEncoder import NumpyEncoder
import json

def write_json(cur_unitData, output_path):
    # # Save as json file
    json_filename = cur_unitData['Unit'] + '_unitData.json'
    with open(output_path + sep + 'JSON files' + sep + json_filename, 'w') as cur_json:
        cur_json.write(json.dumps(cur_unitData, cls=NumpyEncoder, indent=4))