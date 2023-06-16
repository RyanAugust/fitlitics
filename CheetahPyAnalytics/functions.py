import pandas as pd
import numpy as np

class metric_functions:
    def __init__(self):
        self.activity_metric_function_map = {
            'TIZ':   self._a_tiz,
            'IF':    self._a_intensity_factor_power,
            'NP':    self._a_normalized_power,
            'TSS':   self._a_coggan_tss,
            'test':  self._a_test
        }
        self.activity_summary_metric_function_map = {
            'IF':    self._s_intensity_factor_power,
            'TSS':   self._s_coggan_tss

        }
    
    def activity_metric(self, frame: pd.DataFrame, metric_name: str, **kwargs) -> float:
        metric_function = self.activity_metric_function_map[metric_name]
        values = metric_function(frame=frame, **kwargs)
        return values

    def activity_summary_metric(self, frame: pd.DataFrame, metric_name: str, **kwargs) -> pd.Series:
        metric_function = self.activity_summary_metric_function_map[metric_name]
        values = metric_function(frame=frame, **kwargs)
        return values

    def _a_tiz(self, frame: pd.DataFrame, zone_cutoffs: list[int]) -> list[int]:
        """takes input of an activity with power data and zone_delineations
        and returns TIZ for each zone using the passed zone system"""
        tiz_vals = np.array([])
        for z_cutoff in zone_cutoffs:
            tiz = (frame['power'] >= z_cutoff).sum()
            tiz_vals = np.append([tiz_vals, tiz])
        values = tiz_vals - np.append(tiz_vals[1:],[0])
        return values

    def _a_intensity_factor_power(self, frame:pd.DataFrame, FTP:int=None) -> float:
        """Takes input of an activity with power and FTP settings and returns the
        calculated intensity factor"""
        requires = ['power']
        _np = self._a_normalized_power(frame=frame, FTP=FTP)
        value = (_np/frame['power'].mean()).value
        return value

    def _s_intensity_factor_power(self, frame:pd.DataFrame, FTP:int=None) -> pd.Series:
        """Takes input of an activity with power and FTP settings and returns the
        calculated intensity factor"""
        if 'FTP' not in frame.columns:
            assert FTP is not None, "Requires FTP input in dataframe or as parameter"
            values = frame['NP']/FTP
        else:
            values = frame['NP']/frame['FTP']
        return values

    def _a_normalized_power(self, frame:pd.DataFrame, FTP:int=None) -> float:
        """Takes input of an activty with power data and FTP setting and 
        returns the Normalized Power value"""
        _30sr_p = frame['power'].rolling(window=30, min_periods=1)
        value = ( (_30sr_p**4).mean()**(1/4) ).value
        return value

    def _s_coggan_tss(self, frame:pd.DataFrame, FTP:int=None) -> pd.Series:
        """Takes input of an activity summaries with power metrics and FTP settings
        and returns the tss value"""
        required = ['NP','IF','duration']
        if 'FTP' not in frame.columns:
            assert FTP is not None, "Requires FTP input in dataframe or as parameter"
            frame['FTP'] = FTP
        
        values = ((frame['NP']*frame['IF']*frame['duration'])/(frame['FTP']*3600))*100
        return values

    def _a_coggan_tss(self, frame:pd.DataFrame, FTP:int) -> float:
        """Takes input of an activity with power metrics and FTP settings
        and returns the tss value"""
        required = ['power']
        _np = self._a_normalized_power(frame=frame, FTP=FTP)
        _if = self._a_intensity_factor(frame=frame, FTP=FTP)
        _duration = frame.shape[0]
        activity_summary = pd.DataFrame({'NP':[_np], 'IF':[_if], 'duration':[_duration]})

        _tss = self._s_coggan_tss(frame=activity_summary, FTP=FTP)
        return _tss




mets = metric_functions()


class performance_functions(object):
    def __init__(self, athlete_statics):
        # super().__init__()
        self.athlete_statics=athlete_statics
        self.metric_function_map = {
            'VO2':              self.calc_vo2,
            'Garmin VO2':       self.use_garmin_vo2,
            'AE EF':            self.calc_ae_ef,
            'Power Index':      self.use_power_index,
            'Power Index EF':   self.use_power_index_ef,
            'Mod AE Power':     self.modeled_aerobic_threshold_power
        }
    
    def derive_performance(self, frame, performance_metric: str) -> float:
        performance_function = self.metric_function_map[performance_metric]
        values = []
        for index, row in frame.iterrows():
            values.append(performance_function(row))
        frame['performance_metric'] = values
        return frame

    ## TODO: REBUILD TO VECTOR MATH
    # def derive_performance(self, frame: pd.DataFrame, performance_metric: str) -> pd.DataFrame:
    #     performance_function = self.metric_function_map[performance_metric]
    #     frame[performance_metric] = performance_function(frame)
    #     return frame

    def calc_vo2(self, row: pd.Series) -> float:
        try:
            if row['Sport'] == 'Bike':
                percent_vo2 = (row['Average_Heart_Rate'] - self.athlete_statics["resting_hr"])/(self.athlete_statics["max_hr"] - self.athlete_statics["resting_hr"])
                vo2_estimated = (((row['Average_Power']/75)*1000)/row['Athlete_Weight']) / percent_vo2
            elif row['Sport'] == 'Run':
                percent_vo2 = (row['Average_Heart_Rate'] - self.athlete_statics["resting_hr"])/(self.athlete_statics["max_hr"] - self.athlete_statics["resting_hr"])
                vo2_estimated = (210/row['xPace']) / percent_vo2
            else:
                vo2_estimated =  0.0
            return vo2_estimated
        except:
            return 0.0
    ## TODO: REBUILD TO VECTOR MATH
    # def calc_vo2(self, frame):
    #     percent_vo2 = (frame['Average_Heart_Rate'] - self.athlete_statics["resting_hr"])/(self.athlete_statics["max_hr"] - self.athlete_statics["resting_hr"])
    #     bike_vo2_estimated = (((frame['Average_Power']/75)*1000)/frame['Athlete_Weight']) / percent_vo2
    #     run_vo2_estimated = (210/frame['xPace']) / percent_vo2

    #     vo2_dict = {'Bike':bike_vo2_estimated
    #                ,'Run': run_vo2_estimated}
    #     for i,sport in frame['Sport'].unique():
    #         vo2_dict.update({sport:0}) if sport not in vo2_dict.keys() else 0

    #     values = frame['Sport'].apply(lambda sport: vo2_dict[sport])
    #     return values

    def use_garmin_vo2(self, row: pd.Series) -> float:
        vo2_estimated = 0.0
        if (row['Workout_Code'] != 'Rec') & (row['Sport'] in ['Run','Bike']):
            vo2_estimated = row['VO2max_detected'] # Garmin VO2 Estimation
        return vo2_estimated

    def calc_ae_ef(self, row: pd.Series) -> float:
        ef = 0
        if (row['Workout_Code'] == 'AE'):
            if row['Sport'] == 'Bike':
                ef = row['IsoPower']/row['Average_Heart_Rate']
            elif row['Sport'] == 'Run':
                ef = row['IsoPower']/row['Average_Heart_Rate']
        return ef
    
    def use_power_index(self, row: pd.Series) -> float:
        if row['Average_Power'] > 0:
            val = row['Power_Index']
        else:
            val = 0.0
        return val

    def use_power_index_ef(self, row):
        if row['Average_Power'] > 0:
            hr_range = self.athlete_statics['max_hr'] - self.athlete_statics['resting_hr']
            avg_hr_rel = row['Average_Heart_Rate'] - self.athlete_statics['resting_hr']
            relative_hr = (avg_hr_rel / hr_range)*100
            
            pi_ef = row['Power_Index']/relative_hr
            val = pi_ef
        else:
            val = 0
        return val

    def modeled_aerobic_threshold_power(self, row):
        temp = 20
        duration = 60*60
        
        if (row['a'] != 0) & (row['Duration'] > 999):
            power = row['a'] + row['b'] * self.athlete_statics['threshold_hr'] +  row['c'] * duration * temp
            return power
        else:
            return 0
