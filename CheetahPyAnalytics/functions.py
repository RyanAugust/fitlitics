import pandas as pd
import numpy as np

class metric_functions:
    def __init__(self):
        self.activity_metric_function_map = {
            'TIZ':            self._a_tiz,
            'IF':             self._a_intensity_factor_power,
            'NP':             self._a_normalized_power,
            'TSS':            self._a_coggan_tss,
            'VO2':            self._a_calc_vo2,
            'max_power_ef':   self._a_max_Xmin_power_ef,
            'max_power':      self._a_max_Xmin_power,
            'hr_at_power':    self._a_hr_at_power,
            'ae_ef':          self._a_ae_ef

        }
        self.activity_summary_metric_function_map = {
            'IF':    self._s_intensity_factor_power,
            'VO2':   self._s_calc_vo2,
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
            tiz_vals = np.append(tiz_vals, tiz)
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
        value = ((_30sr_p**4).mean()**(1/4)).value
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

    def _a_coggan_tss(self, frame:pd.DataFrame, FTP:int, **kwargs) -> float:
        """Takes input of an activity with power metrics and FTP settings
        and returns the tss value"""
        required = ['power']
        _np = self._a_normalized_power(frame=frame, FTP=FTP)
        _if = self._a_intensity_factor(frame=frame, FTP=FTP)
        _duration = frame.shape[0]
        activity_summary = pd.DataFrame({'NP':[_np], 'IF':[_if], 'duration':[_duration]})

        _tss = self._s_coggan_tss(frame=activity_summary, FTP=FTP)
        return _tss
    
    def _a_calc_vo2(
        self,
        frame:pd.DataFrame,
        sport:str,
        resting_hr:int,
        max_hr:int,
        athlete_mass:float) -> float:
        """Takes input of an activity with power and hr data and returns an
        estimated VO2max value"""
        _hr = frame['heart_rate'].mean()
        _pct_VO2 = (_hr - resting_hr)/(max_hr - resting_hr)

        if sport == 'Bike':
            _eff = (frame['power'].mean()/75)*1000/athlete_mass
        elif sport == 'Run':
            _eff = 210/frame['pace'].mean()
        value = _eff / _pct_VO2
        return value

    def _s_calc_vo2(
        self,
        frame:pd.DataFrame, 
        resting_hr:int=None,
        max_hr:int=None,
        athlete_mass:float=None,
        **kwargs) -> float:
        """Takes input of an activity summary with power, heart rate, and sport data 
        and returns estimated VO2max values"""
        param_data = {'resting_hr':resting_hr,
            'max_hr':max_hr,
            'athlete_weight':athlete_weight}
        for metric_name, metric_field in param_data.items():
            # there has to be a better way to do this.... RESEARCH
            if metric_name not in frame.columns:
                assert metric_field is not None, f"Requires {metric_name} input in dataframe or as parameter"
            else:
                param_data[metric_name] = frame[metric_name]

        _pct_VO2 = ((frame['average_heart_rate'] - param_data['resting_hr']) /
                    (param_data['max_hr'] - param_data['resting_hr']))

        _eff = np.where(frame['sport']=='Bike',
                    (frame['power'].mean()/75)*1000/param_data['athlete_mass'],
                    210/frame['pace'].mean())
        values = _eff / _pct_VO2
        return values

    def _a_max_Xmin_power_ef(
        self,
        frame:pd.DataFrame,
        power_duration:int) -> float:
        rolling_power = frame['power'].rolling(window=power_duration).mean()
        max_power_idx = rolling_power.argmax()
        max_power = rolling_power.max()
        max_power_hr = frame['heart_rate'][max_power_idx-power_duration:max_power_idx].mean()
        value = max_power / max_power_hr
        return value

    def _a_max_Xmin_power(
        self,
        frame:pd.DataFrame,
        power_duration:int) -> float:
        value = frame['power'].rolling(window=power_duration).mean().max()
        return value
    
    def _a_hr_at_power(
        self,
        frame:pd.DataFrame,
        find_power_level:int,
        power_duration:int=300,
        tol:float=5) -> float:
        """Takes input of an activity, a desired power level, power duration,
        and a tolerance. Power values are taken at a 10-second rolling average and
        tested against the assigned parameters
        Returns a value for the heart rate sustainded during"""
        rolling_power = frame['power'].rolling(window=10).mean()
        u_rolling_power = rolling_power > find_power_level - tol
        l_rolling_power = rolling_power < find_power_level + tol
        rp_win_tol = u_rolling_power + l_rolling_power
        rp_win_tol_idx = (rp_win_tol.rolling(window=power_duration) > power_duration * 2).argmin()
        # I guess I should take the argmin to get the first occurance of stable power
        value = frame['heart_rate'][rp_win_tol_idx-power_duration:rp_win_tol_idx].mean()
        return value

    def _a_ae_ef(
            self,
            frame:pd.DataFrame,) -> float:
        ef = np.where(frame['sport'] in ['Bike','Run'],
                      frame['IsoPower']/frame['Average_Heart_Rate'],
                      np.nan)
        return ef


    # def modeled_aerobic_threshold_power(self, row):
    #     temp = 20
    #     duration = 60*60
        
    #     if (row['a'] != 0) & (row['Duration'] > 999):
    #         power = row['a'] + row['b'] * self.athlete_statics['threshold_hr'] +  row['c'] * duration * temp
    #         return power
    #     else:
    #         return 0
