from re import split, search, match, finditer
from glob import glob
from datetime import datetime
from platform import system
from os.path import sep
# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep
import json

from pandas import DataFrame, read_csv


def extract_subject_and_date(unit_id):
    """
    Robustly extract subject_id and cur_date from a unit_id string, tolerant of:
    - duplicated subject-ID fields (e.g. SUBJ-ID-1219_SUBJ-ID-1219_...)
    - 2-digit-year dates (YYMMDD or YY-MM-DD)
    - 4-digit-year dates (YYYY-MM-DD)

    Locates the date by pattern-matching each underscore-delimited token rather
    than assuming a fixed positional index, so extra/duplicated fields (like a
    repeated subject ID) don't shift which token gets misread as the date.
    """
    tokens = split('_*_', unit_id)

    # Subject ID is always the first token (SUBJ-ID-XXXX pattern)
    subject_id = tokens[0]

    # Accepts: YYMMDD, YY-MM-DD, YYYY-MM-DD
    date_patterns = [
        (r'^\d{6}$', '%y%m%d'),
        (r'^\d{2}-\d{2}-\d{2}$', '%y-%m-%d'),
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
    ]

    cur_date = None
    date_format = None
    for tok in tokens:
        for pattern, fmt in date_patterns:
            if match(pattern, tok):
                cur_date = tok
                date_format = fmt
                break
        if cur_date is not None:
            break

    if cur_date is None:
        raise ValueError(f"Could not locate a date token in unit_id: {unit_id!r}")

    return subject_id, cur_date, date_format


def extract_session_key(filename):
    """Extract the date+time substring that uniquely identifies a recording
    session from a behavioral (trialInfo/spoutTimestamps/optoTimestamps)
    filename, regardless of its position or how many other fields precede it.

    Handles both known formats by scanning for date/time-shaped chunks
    anywhere in the string (order determines which is date vs time — date
    always precedes time in this filenaming convention):
      - compact 6-digit runs, e.g. 'MML-Aversive-AM-210501-112033' (Synapse: YYMMDD-HHMMSS)
      - dashed groups, e.g. '2021-07-17_15-19-28' (Intan: YYYY-MM-DD then HH-MM-SS)

    Args:
        filename (str): A trialInfo/spoutTimestamps/optoTimestamps filename
            (or just its stem) containing an embedded date and time.

    Returns:
        str: The literal substring spanning from the start of the first
        (date) match through the end of the second (time) match, suitable
        for use as a regex search pattern against this session's sibling files.

    Raises:
        ValueError: If fewer than two date/time-shaped tokens are found.
    """
    pattern = r'(?<!\d)(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{2}|\d{6})(?!\d)'
    matches = list(finditer(pattern, filename))
    if len(matches) < 2:
        raise ValueError(f"Could not find both a date and a time token in filename: {filename!r}")
    return filename[matches[0].start():matches[1].end()]


def match_spike_times_with_keys(input_list):
    """Locate the behavioral key files matching a spike-times memory file, and
    load or initialize that unit's JSON data record.

    Args:
        input_list (tuple): ``(memory_path, all_json, SETTINGS_DICT)`` where
            ``memory_path`` is the path to the unit's spike-times file,
            ``all_json`` is a list of existing unit-data JSON paths (used to
            resume a previously created record instead of starting fresh),
            and ``SETTINGS_DICT`` holds pipeline configuration (recording
            type/sampling-rate lookups, KEYS_PATH, BREAKPOINT_PATH, etc).

    Returns:
        tuple: ``(memory_path, key_paths_info, key_paths_spoutTTL,
        key_paths_optoTTL, cur_unitData, cur_breakpoint_df)``. Returns None
        (no tuple) if no matching trialInfo key file could be found for this
        unit — callers must handle that case explicitly since unpacking None
        will raise.
    """
    # Match spike_times with appropriate key_files

    memory_path, all_json, SETTINGS_DICT = input_list
    # Split path name to get subject, session and unit ID for prettier output
    split_memory_path = split(REGEX_SEP, memory_path)  # split path
    unit_id = split_memory_path[-1][:-4]  # Example unit id: SUBJ-ID-26_200711_concat_cluster41

    subject_id, cur_date, cur_date_format = extract_subject_and_date(unit_id)

    recording_type = SETTINGS_DICT['RECORDING_TYPE_DICT'][subject_id]
    sampling_rate = SETTINGS_DICT['SAMPLING_RATE_DICT'][recording_type]

    # Normalize cur_date to both legacy formats up front, so downstream globs
    # can try each explicitly instead of guessing via exception control flow.
    parsed_date = datetime.strptime(cur_date, cur_date_format)
    date_yymmdd  = datetime.strftime(parsed_date, '%y%m%d')
    date_yy_dash = datetime.strftime(parsed_date, '%y-%m-%d')

    # Use subj-session identifier to grab appropriate key
    # Stimulus info is in trialInfo

    # These are in alphabetical order. Must sort by date_trial or match with file
    # Match by name for now for breakpoints
    key_paths_info = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                          date_yy_dash + '*_trialInfo.csv')
    key_paths_spoutTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                           date_yy_dash + '*spoutTimestamps.csv')
    key_paths_optoTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                           date_yy_dash + '*optoTimestamps.csv')

    if len(key_paths_info) == 0:
        # Try the other legacy format (Intan-style, no dashes)
        key_paths_info = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                              date_yymmdd + '*_trialInfo.csv')
        key_paths_spoutTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                               date_yymmdd + '*_spoutTimestamps.csv')
        key_paths_optoTTL = glob(SETTINGS_DICT['KEYS_PATH'] + sep + subject_id + '*' +
                               date_yymmdd + '*_optoTimestamps.csv')

    if len(key_paths_info) == 0:
        print('Key not found for ' + unit_id)
        return

    # Try both normalized date forms for the breakpoint file, no exception-driven guessing
    cur_breakpoint_file_matches = glob(SETTINGS_DICT['BREAKPOINT_PATH'] + sep + subject_id + '*' +
                                       date_yy_dash + '*' + '_breakpoints.csv')
    if len(cur_breakpoint_file_matches) == 0:
        cur_breakpoint_file_matches = glob(SETTINGS_DICT['BREAKPOINT_PATH'] + sep + subject_id + '*' +
                                           date_yymmdd + '*' + '_breakpoints.csv')

    if len(cur_breakpoint_file_matches) == 0:
        print('Breakpoint file not found for ' + unit_id + '. Assuming non-concatenated...')
        cur_breakpoint_df = DataFrame()
    else:
        cur_breakpoint_file = cur_breakpoint_file_matches[0]
        cur_breakpoint_df = read_csv(cur_breakpoint_file)

    # If no JSON for that unit exists, create UnitData
    try:
        cur_unitData_name = all_json[
            all_json.index(SETTINGS_DICT['OUTPUT_PATH'] + sep + 'JSON files' + sep + unit_id + '_unitData.json')]
        with open(cur_unitData_name, 'r') as json_file:
            cur_unitData = json.load(json_file)
    except ValueError:
        cur_unitData = {'Unit': unit_id, 'Subject': subject_id,
                        'Recording_type': recording_type,
                        'Sampling_rate': sampling_rate,
                        'Session': {}}

    return memory_path, key_paths_info, key_paths_spoutTTL, key_paths_optoTTL, cur_unitData, cur_breakpoint_df


def process_metadata(subject_id, key_path_info, key_paths_spoutTTL, key_paths_optoTTL,
                     cur_breakpoint_df, sampling_rate):
    '''
    Grab some files and specifics about each behavioral file.

    Derives a "key_finder" substring from the trialInfo filename via
    ``extract_session_key`` (content-based date+time detection, independent
    of recording type or filename layout) and uses it to locate the matching
    spout-TTL and opto-TTL files, plus this session's breakpoint offset (for
    concatenated recordings) carried over from the previous session in the
    breakpoint table.

    Args:
        subject_id (str): Subject identifier, used only for log messages.
        key_path_info (str): Path to this session's trialInfo CSV.
        key_paths_spoutTTL (list[str]): Candidate spout-TTL file paths to search.
        key_paths_optoTTL (list[str]): Candidate opto-TTL file paths to search.
        cur_breakpoint_df (pandas.DataFrame): Breakpoint table for this
            subject/date; empty if no breakpoint file was found upstream.
        sampling_rate (float): Used to convert legacy 'Break_point' (samples)
            to seconds when 'Break_point_seconds' isn't available.

    Returns:
        tuple: (key_path_spoutTTL, key_path_optoTTL, breakpoint_offset,
        key_finder). Either TTL path may be None if not found (spout is
        expected to be missing for passive recordings; opto only when opto
        was performed). breakpoint_offset is 0 for the first file in a
        concatenated recording (no prior session to offset from).
    '''
    key_finder = extract_session_key(split(REGEX_SEP, key_path_info)[-1])

    try:
        spoutTTL_path_finder = [search(key_finder, file_name) for file_name in key_paths_spoutTTL]
        spoutTTL_path_finder = [i for i, x in enumerate(spoutTTL_path_finder) if x is not None][0]
        key_path_spoutTTL = key_paths_spoutTTL[spoutTTL_path_finder]
    except IndexError:
        print('Spout TTL file not found for ' + subject_id + ', file ' + key_path_info+ '. Ignore if this is a passive recording.')
        key_path_spoutTTL = None

    try:
        optoTTL_path_finder = [search(key_finder, file_name) for file_name in key_paths_optoTTL]
        optoTTL_path_finder = [i for i, x in enumerate(optoTTL_path_finder) if x is not None][0]
        key_path_optoTTL = key_paths_optoTTL[optoTTL_path_finder]
    except IndexError:
        print('Opto TTL file not found for ' + subject_id + ', file ' + key_path_info+ '. Ignore if no opto was done.')
        key_path_optoTTL = None

    # Find appropriate breakpoint for file if it exists
    try:
        breakpoint_offset_idx = cur_breakpoint_df.index[
            cur_breakpoint_df['Session_file'].str.contains(key_finder)]
    except KeyError:
        print('Breakpoint file not found for ' + subject_id + ', file ' + key_path_info + '. Ignore if this is a non-cocatenated recording.')
        breakpoint_offset_idx = 0

    # also grab previous session's breakpoint if it exists
    try:
        breakpoint_offset = cur_breakpoint_df.loc[
            breakpoint_offset_idx - 1, 'Break_point_seconds'].values[0]
    # Older recordings do not have Break_point_seconds but Break_point. Need to divide by sampling rate
    except KeyError:
        try:
            breakpoint_offset = cur_breakpoint_df.loc[
                breakpoint_offset_idx - 1, 'Break_point'].values[0]  # grab previous session's breakpoint
            breakpoint_offset = breakpoint_offset / sampling_rate
        except IndexError as e:
            print('Something off with ' + subject_id + ', file ' + key_finder)
            raise e
        except KeyError:
            breakpoint_offset = 0  # first file; no breakpoint offset needed
        except Exception as e:
            raise e
    return key_path_spoutTTL, key_path_optoTTL, breakpoint_offset, key_finder