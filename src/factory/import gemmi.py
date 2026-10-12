import gemmi
import pandas as pd


def read_mtz_to_data_frame(file_path: string) -> pd.DataFramr

    mtz = gemmi.read_mtz_file(file_path)

    _mtz_data = { col.label: col.array for col in mtz.columns }
    mtz_data = pd.DataFrame(_mtz_data)
    print("mtz loaded successfully")
    print(mtz_data.head())
    return mtz_data
