import json
import json_stream
import pandas as pd

def get_JSON_data(json_filenames, sessions_to_run=None, sessions_to_exclude=None, fields_to_exclude=None):
    """
    :param stream_JSON:
    :param json_filenames: list of paths
    :param sessions_to_run: looks for substrings in the JSON filenames and allows
    :param sessions_to_exclude: looks for substrings in the JSON filenames and excludes
    :return: [unit_list, data_list]:
                unit_list: list of unit names
                data_list: list of all data present in JSON file
    """
    data_list = []
    if type(sessions_to_run) == str:  # file path
        sessions_file = pd.read_csv(sessions_to_run)
        sessions_to_run = set(sessions_file['Unit'].values)

    for file_name in json_filenames:
        if sessions_to_run is not None:
            if any([chosen for chosen in sessions_to_run if chosen in file_name]):
                pass
            else:
                continue

        if sessions_to_exclude is not None:
            if any([chosen for chosen in sessions_to_exclude if chosen in file_name]):
                continue
            else:
                pass


        # Open JSON
        with open(file_name, 'r') as json_file:
            cur_dict = json.load(json_file)
        if fields_to_exclude is not None:
            for field in fields_to_exclude:
                cur_dict.pop(field, None)

        data_list.append(cur_dict)

    return data_list
