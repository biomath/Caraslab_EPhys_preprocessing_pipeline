import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# def _load_one_json(file_name, fields_to_exclude=None):
#     with open(file_name, "r", encoding="utf-8") as json_file:
#         cur_dict = json.load(json_file)
#
#     if fields_to_exclude is not None:
#         for field in fields_to_exclude:
#             cur_dict.pop(field, None)
#
#     return cur_dict

# Obsolete
# def get_JSON_data(
#     json_filenames,
#     sessions_to_run=None,
#     sessions_to_exclude=None,
#     fields_to_exclude=None,
#     max_workers=8,
# ):
#     if isinstance(sessions_to_run, str):
#         sessions_file = pd.read_csv(sessions_to_run)
#         sessions_to_run = set(sessions_file["Unit"].values)
#
#     fields_to_exclude = set(fields_to_exclude) if fields_to_exclude is not None else None
#
#     filtered_files = []
#     for file_name in json_filenames:
#         if sessions_to_run is not None and not any(
#             chosen in file_name for chosen in sessions_to_run
#         ):
#             continue
#
#         if sessions_to_exclude is not None and any(
#             chosen in file_name for chosen in sessions_to_exclude
#         ):
#             continue
#
#         filtered_files.append(file_name)
#
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         data_list = list(
#             executor.map(
#                 lambda fn: _load_one_json(fn, fields_to_exclude),
#                 filtered_files,
#             )
#         )
#
#     return data_list


def _load_one_json(file_name, fields_to_exclude=None):
    """Read and parse a single unit-data JSON file, skipping empty/corrupt files.

    Called per-file (e.g. from ``zscore_timeSeries_fromJSON.py``) rather than
    from ``get_JSON_data`` itself, so files can be loaded lazily one at a time
    instead of all at once.

    Args:
        file_name (str): Path to the JSON file to load.
        fields_to_exclude (Iterable[str], optional): Top-level keys to drop
            from the parsed dict before returning it.

    Returns:
        dict or None: Parsed JSON contents, or None if the file was empty or
            failed to parse (a warning is printed to stdout in that case).
    """
    with open(file_name, "r", encoding="utf-8") as f:
        raw = f.read()

    if not raw.strip():
        print(f"Warning: empty file skipped: {file_name}")
        return None

    try:
        cur_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in {file_name}: {e}")
        return None

    if fields_to_exclude is not None:
        for field in fields_to_exclude:
            cur_dict.pop(field, None)

    return cur_dict

def get_JSON_data(
    json_filenames,
    sessions_to_run=None,
    sessions_to_exclude=None):
    """Filter a list of unit-data JSON paths down to the sessions of interest.

    Note this only filters filenames — it does not load file contents. Load
    each returned path individually with ``_load_one_json`` when needed.

    Args:
        json_filenames (Iterable[str]): Candidate JSON file paths to filter.
        sessions_to_run (str or Iterable[str], optional): Either a CSV path
            with a "Unit" column, or an iterable of substrings — only paths
            containing one of these are kept. If None, no inclusion filter
            is applied.
        sessions_to_exclude (Iterable[str], optional): Substrings whose
            presence in a path excludes it, applied after ``sessions_to_run``.

    Returns:
        list[str]: Filtered list of JSON file paths (unloaded).
    """
    if isinstance(sessions_to_run, str):
        sessions_file = pd.read_csv(sessions_to_run)
        sessions_to_run = set(sessions_file["Unit"].values)

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

    return filtered_files  # just paths, no loading