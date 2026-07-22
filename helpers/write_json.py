from os import sep
from helpers.NumpyEncoder import NumpyEncoder
import json


def write_json(cur_data, output_path, json_filename):
    """Serialize a dict-like structure to a JSON file, tolerating numpy types.

    Args:
        cur_data: Data to serialize (may contain numpy scalars/arrays via NumpyEncoder).
        output_path (str): Directory to write the file into.
        json_filename (str): Name of the output JSON file.
    """
    # Serialize first — if this fails, the file is never touched
    serialized = json.dumps(cur_data, cls=NumpyEncoder, indent=4)

    with open(output_path + sep + json_filename, 'w') as cur_json:
        cur_json.write(serialized)