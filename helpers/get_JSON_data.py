import json
import pandas as pd

def get_JSON_data(json_filenames, session_type, sessions_to_run=None, sessions_to_exclude=None):
    """
    :param json_filenames: list of paths
    :param session_type: 'pre', 'active', 'post' or 'post1h'; specific to experiment naming in Synapse and Intan
    :param sessions_to_run: looks for substrings in the JSON filenames and allows
    :param sessions_to_exclude: looks for substrings in the JSON filenames and excludes
    :return: [unit_list, data_list]:
                unit_list: list of unit names
                data_list: list of all data present in JSON file
    """
    unit_list = []
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

        # Grab  data
        session_names = list(cur_dict['Session'].keys())
        if session_type == 'active':
            try:
                session_name = [s for s in session_names if ("Aversive" in s) or ("Active" in s)][0]
            except IndexError:
                continue
        elif session_type == 'pre':
            try:
                session_name = [s for s in session_names if ("Pre" in s)][0]
            except IndexError:
                continue

        elif session_type == 'post':
            try:
                session_name = [s for s in session_names if ("Post_" in s) or ("Post-" in s)][0]
            except IndexError:
                continue
        elif session_type == 'post1h':
            try:
                session_name = [s for s in session_names if ("Post1h" in s)][0]
            except IndexError:
                continue
        else:
            print('Session name undefined')
            exit()

        cur_data = cur_dict['Session'][session_name]
        cur_data['Session'] = session_name

        data_list.append(cur_data)
        unit_list.append(cur_dict['Unit'])

    return [unit_list, data_list]
