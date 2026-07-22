from os import remove, sep
from glob import glob
from pandas import read_csv, concat

def compile_fr_result_csv(master_sheet_name, output_path, overwrite_previous):
    """Merge per-worker temp CSVs produced by parallel FR calculation into one master CSV.

    Each parallel worker writes its own rows to a
    ``*_tempfile_<master_sheet_name>`` file (see ``run_ephys_pipeline.py``);
    this stitches them all into ``master_sheet_name`` and deletes the temp files.

    Args:
        master_sheet_name (str): Filename of the compiled master CSV.
        output_path (str): Directory containing the temp files and the master CSV.
        overwrite_previous (bool): If True, (re)write the master CSV with a header
            (overwriting any existing file). If False, rows are only appended, so
            the master CSV must already exist with a header from a prior run —
            otherwise the appended rows will lack a header row.
    """
    process_files = glob(output_path + sep + '*_tempfile_' + master_sheet_name)

    # Read first process csv just to get the header
    df_header = read_csv(process_files[0], nrows=0)

    # Now read all process csv to compile
    df_merged = (read_csv(f, sep=',', header=None, skiprows=1) for f in process_files)
    df_merged = concat(df_merged, ignore_index=True)

    if overwrite_previous:
        df_header.to_csv(output_path + sep + master_sheet_name, mode='w', header=True, index=False)

    df_merged.to_csv(output_path + sep + master_sheet_name, mode='a', header=False, index=False)

    [remove(f) for f in process_files]
