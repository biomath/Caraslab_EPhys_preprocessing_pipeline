from os import remove, sep
from glob import glob
from pandas import read_csv, concat

def compile_fr_result_csv(csv_prename, output_path, overwrite_previous):
    master_sheet_name = csv_prename + '_AMsound_firing_rate.csv'

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
