import os
import pandas as pd
import numpy as np
import json

class open_dataset(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.athlete_ids = self.get_athlete_ids()

    def get_athlete_ids(self):
        for a,b,c in os.walk(self.root_dir):
            athletes = b
            athletes.remove('INDEX') if 'INDEX' in athletes else 0
            break
        return athletes

    def show_athlete_ids(self):
        athlete_id_count = len(self.athlete_ids)
        athlete_ids_joined = ",\n".join(self.athlete_ids)
        ath_id_str = f"AVAILABLE ATHLETE IDS ({athlete_id_count}): \n {athlete_ids_joined}"
        print(ath_id_str)

    def get_athlete_summary(self, athlete_id, make_float = True):
        ath_summary_path = self._athlete_summary_path(athlete_id=athlete_id)
        with open(ath_summary_path, 'r') as f:
            summary_json = f.read()
        f.close()
        rides = json.loads(summary_json)['RIDES']
        df = pd.json_normalize(rides)
        if make_float:
            for col in df.columns.tolist():
                if 'METRIC' in col:
                    if type(df[col].dropna().values[0]) == str:
                        df[col] = self._safe_convert(original_series=df[col], type_convert=float)
                    elif type(df[col].dropna().values[0]) == list:
                        try:
                            decompression = self._safe_list_decompression(original_series=df[col], type_convert=float)
                            df = df.join(decompression)
                            del df[col]
                        except:
                            print(f'{col}--fail')
                    else:
                        None
        return df
    
    def get_athlete_activity_files(self, athlete_id: str) -> list:
        athlete_dir = self._athlete_dir(athlete_id=athlete_id)
        for a, b, c in os.walk(athlete_dir):
            raw_files = c
        activity_files = []
        [activity_files.append(file) if '.csv' in file else 0  for file in raw_files]
        return activity_files
    
    def _athlete_dir(self, athlete_id: str) -> str:
        athlete_dir = os.path.join(self.root_dir, athlete_id)
        return athlete_dir

    def _athlete_summary_path(self, athlete_id: str)  -> str:
        summary_filename = "{" + athlete_id + "}.json"
        ath_summary_path = os.path.join(self.root_dir, athlete_id, summary_filename)
        return ath_summary_path
    
    @staticmethod
    def _safe_convert(original_series: pd.Series, type_convert: type) -> pd.Series:
        try:
            new_series = original_series.astype(type_convert)
            return new_series
        except:
            return original_series
    @staticmethod
    def _safe_list_decompression(original_series: pd.Series,  type_convert: type) -> pd.DataFrame:
        metric_base_name = original_series.name
        def safe_split(val):
            if type(val) == list:
                return float(val[0]), float(val[1])
            elif type(val) == str:
                return float(val)
            else:
                return 0, 0
        slim_original_series = original_series.dropna()
        decompressed_df = pd.DataFrame(original_series.dropna().tolist(), index=slim_original_series.index)
        if decompressed_df.shape[1] == 2:
            decompressed_df.columns = [f'{metric_base_name}_value',f'{metric_base_name}_duration']
        else:
            decompressed_df.columns = [f'{metric_base_name}_value_{x}' for x in range(decompressed_df.shape[1])]
        return decompressed_df

