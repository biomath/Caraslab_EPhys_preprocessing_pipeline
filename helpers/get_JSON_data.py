import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def _load_one_json(file_name, fields_to_exclude=None):
    with open(file_name, "r", encoding="utf-8") as json_file:
        cur_dict = json.load(json_file)

    if fields_to_exclude is not None:
        for field in fields_to_exclude:
            cur_dict.pop(field, None)

    return cur_dict

def get_JSON_data(
    json_filenames,
    sessions_to_run=None,
    sessions_to_exclude=None,
    fields_to_exclude=None,
    max_workers=8,
):
    if isinstance(sessions_to_run, str):
        sessions_file = pd.read_csv(sessions_to_run)
        sessions_to_run = set(sessions_file["Unit"].values)

    fields_to_exclude = set(fields_to_exclude) if fields_to_exclude is not None else None

    filtered_files = []
    for file_name in json_filenames:
        if sessions_to_run is not None and not any(
            chosen in file_name for chosen in sessions_to_run
        ):
            continue

        if sessions_to_exclude is not None and any(
            chosen in file_name for chosen in sessions_to_exclude
        ):
            continue

        filtered_files.append(file_name)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list = list(
            executor.map(
                lambda fn: _load_one_json(fn, fields_to_exclude),
                filtered_files,
            )
        )

    return data_list
