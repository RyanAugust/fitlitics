import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from cheetahpy import CheetahPy


class fetch_new_dataset:
    def __init__(self):
        self.metrics_list = ['Duration', 'TSS', 'Average_Heart_Rate', 'Max_Heartrate', 
            'Average_Power', 'Athlete_Weight', 'Estimated_VO2MAX', '10_sec_Peak_Pace_Swim', 'xPace', 
            'Pace', 'IsoPower', 'Power_Index', 'L1_Time_in_Zone', 'L2_Time_in_Zone', 'L3_Time_in_Zone', 
            'L4_Time_in_Zone', 'L5_Time_in_Zone', 'L6_Time_in_Zone', 'L7_Time_in_Zone']
        self.metadata_list = ['VO2max_detected', 'Shoes', 'Workout_Code', 'Workout_Title', 
            'Indoor', 'Frame', 'Sport']
    
    def build_gc_request(self):                                                         ## TODO: rebuild using CheetahPy
        base_api_endpoint = 'http://localhost:12021/Ryan%20Duecker?metrics={metrics_fields}&metadata={metadata_fields}'
        fmted_endpoint = base_api_endpoint.format(metrics_fields=','.join(self.metrics_list),
                                                  metadata_fields=','.join(self.metadata_list))
        return fmted_endpoint
    
    def build_new_dataset(self):
        data_original = pd.read_csv(
            self.build_gc_request()
        )
        data_original.columns = [x.strip(' "') for x in data_original.columns]
    
        data_original['date'] = pd.to_datetime(data_original['date'])
        data_original['VO2max_detected'] = data_original['VO2max_detected'].astype(float)
        
        self.save_dataframe(data_original, name='gc_activitydata_local')

        ## Set list of activities from earlier filtered call
        self.activity_filenames = data_original[data_original['Average_Power']>0]['filename'].tolist()
    
    def calculate_activity_ef_params(self, update:bool=False):
        all_filenames = self.activity_filenames
        if update:
            try:
                old_file = pd.read_csv('modeled_ef.csv')
                for file in old_file['files']:
                    all_filenames.remove(file)
                files_modeled = self.process_filenames(file_list=all_filenames)
            except Exception as exc:
                update = False
                print(f"{exc} --couldn't load previous modeled_ef dataset")
                ## model ef
                files_modeled = self.process_filenames()
        df = pd.DataFrame(files_modeled['modeled'],
            files_modeled['files']).reset_index()
        df.columns = ['files','a','b','c','rmse']
        if update:
            df = pd.concat([old_file,df])

        self.save_dataframe(df, name='modeled_ef')

    def save_dataframe(self, df:pd.DataFrame, name:str, dir:str='./', index_save_status=False):
        save_path = os.path.join(dir,f'{name}.csv')
        df.to_csv(save_path, index=index_save_status)
        print(f'{name} saved')

    def extract_activity_data(self, filename:str):
        ## Load gc api module to access individual activities 
        ac = CheetahPy.get_activity(athlete="Ryan Duecker",
            activity_filename=filename)
        var_Ti = np.where(ac['temp'].mean() < -20, 20, ac['temp'].mean())
        var_HRi = ac['hr'].to_numpy()
        var_PWRi = ac['watts'].to_numpy()
        var_t = ac['secs'].to_numpy()
        cons_lag = 15

        ## Genral Formula
        # P_it = a_i + b_i*H_i,t+l + c*t*T_i
        X = np.vstack((var_HRi[cons_lag:],(var_t[:-cons_lag] * var_Ti))).T
        y = var_PWRi[:-cons_lag]
        return X, y

    def make_coef(self, X,y):
        reg = LinearRegression(fit_intercept=True).fit(X, y)
        a = reg.intercept_
        b,c = reg.coef_
        rmse = np.sqrt(((y - reg.predict(X))**2).mean())
        return a,b,c, rmse

    def process_filenames(self, file_list=[]):
        if file_list == []:
            file_list = self.activity_filenames

        details = {'files':file_list,'modeled':[]}
        total_fns = len(file_list)
        for i, fn in enumerate(file_list):
            if i % 25 == 0:
                print("{}/{} activities modeled".format(i,total_fns))
            X, y = self.extract_activity_data(fn)
            a,b,c, rmse = self.make_coef(X,y)
            details['modeled'].append([a,b,c, rmse])
        return details

class dataset_preprocess:
    def __init__(self, local_activity_store=None, local_activity_model_params=None, athlete_statics=static_metrics):
        self.athlete_statics = athlete_statics
        self.local_activity_store = local_activity_store
        self.local_activity_model_params = local_activity_model_params

        if local_activity_store is not None:
            self.activity_data = self.load_dataset(local_activity_store)
        if local_activity_model_params is not None:
            self.modeled_data = self.load_dataset(local_activity_model_params)

    def load_dataset(self, filepath):
        data = pd.read_csv(filepath)
        return data

    def load_local_activity_store(self, filepath):
        self.activity_data = self.load_dataset(filepath)
        self.activity_data['date'] = pd.to_datetime(self.activity_data['date'])
    
    def load_local_activity_model_params(self, filepath):
        self.modeled_data = self.load_dataset(filepath)

    def power_index_maker(self, power, duration, cp=340, w_prime=15000, pmax=448):
        theoretical_power = w_prime/duration - w_prime/(cp-pmax) + cp
        power_index = (power/theoretical_power)*100
        return power_index
        
    def _filter_absent_data(self):
        self.activity_data = self.activity_data[~(((self.activity_data['Sport'] == 'Run') 
                                                    & (self.activity_data['Pace'] <= 0))
                                                | ((self.activity_data['Sport'] == 'Bike') 
                                                    & (self.activity_data['Average_Power'] <= 0))
                                                | (self.activity_data['Average_Heart_Rate'] <= 0))].copy()
        return 0

    def _reframe_data_tss(self):
        self.activity_data.rename(columns={'date':'workoutDate'}, inplace=True)
        ## transform doesn't compress the frame and instead matches index to index
        self.activity_data['day_TSS'] = self.activity_data['TSS'].groupby(
            self.activity_data['workoutDate']).transform('sum').fillna(0)
        return 0
    
    def _prune_relative_to_performance_metric(self, performance_lower_bound):
        # self.activity_data['performance_metric'] = np.where(self.activity_data['Duration'] < 60*60, 0, self.activity_data['performance_metric'])
        self.activity_data['performance_metric'] = np.where(
            self.activity_data['performance_metric'] < performance_lower_bound,
            0,
            self.activity_data['performance_metric'])
        self.activity_data['performance_metric'] = self.activity_data['performance_metric'].replace(0,np.nan)
        self.activity_data['performance_metric'] = self.activity_data['performance_metric'].fillna(method='ffill')
        return 0
    
    def impute_dates(self, fill_performance_forward=False):
        self.processed_activity_data.reset_index(inplace=True)

        self.processed_activity_data = self.processed_activity_data.sort_values(by=['date'])

        self.processed_activity_data.index = pd.DatetimeIndex(self.processed_activity_data['date'])
        missing_dates = pd.date_range(start=self.processed_activity_data.index.min(),
            end=self.processed_activity_data.index.max())
        try:
            self.processed_activity_data = self.processed_activity_data.reindex(missing_dates, fill_value=0)
        except Exception as exc:
            print(f'{exc} occured. Running deduplication and retrying')
            self.processed_activity_data = self.processed_activity_data[~self.processed_activity_data.index.duplicated()]
            self.processed_activity_data = self.processed_activity_data.reindex(missing_dates, fill_value=0)
        
        # drop extra (incomplete) date col
        self.processed_activity_data = self.processed_activity_data[['load_metric','performance_metric']]

        # Fill missing performance data
        if fill_performance_forward:
            self.processed_activity_data['performance_metric'] = self.processed_activity_data['performance_metric'].replace(0,np.nan)
            self.processed_activity_data['performance_metric'] = self.processed_activity_data['performance_metric'].fillna(method='ffill')
            self.processed_activity_data = self.processed_activity_data.dropna()
        return 0


    def pre_process(self, load_metric:str, performance_metric:str, performance_lower_bound:float=0.0, 
                    sport:bool=False, filter_sport:list=[], fill_performance_forward:bool=True) -> str:
        self._filter_absent_data()
        if filter_sport != []:
            self.activity_data = self.activity_data[self.activity_data['Sport'].isin(filter_sport)]

        ## Use identified fxn to create load metric for activity row
        lfxs = load_functions()
        self.activity_data = lfxs.derive_load(frame=self.activity_data, load_metric=load_metric)

        ## Use identified fxn to create performace metric for activity row
        pfxns = performance_functions(athlete_statics=self.athlete_statics)
        self.activity_data = pfxns.derive_performance(frame=self.activity_data, performance_metric=performance_metric)

        ## prune frame based of performance metric
        self._prune_relative_to_performance_metric(performance_lower_bound=performance_lower_bound)

        ## Aggregate frame to daily (+ sport data)
        agg_dict = {'load_metric':'sum','performance_metric':'max'}
        groupby_list = ['date']
        if sport:
            groupby_list.append('Sport')
        self.processed_activity_data = self.activity_data.groupby(groupby_list).agg(agg_dict)
        
        # Impute missing dates to create daily values + handle performance data
        self.impute_dates(fill_performance_forward=fill_performance_forward)

        return "pre-process successful"


# if __name__ == "__main__":
#     dataset_loader = dataset()
#     print("Building new activity dataset")
#     dataset_loader.build_new_dataset()
#     print("Building new ef coef dataset")
#     dataset_loader.calculate_activity_ef_params()
#     print('Done!')
# else:
#     try:
#         os.path.getsize('gc_activitydata_local.csv')
#         print('Local dataset available. process using `dataset_preprocess`')
#     except:
#         dataset_loader = dataset()
#         print('No local dataset available. build using `dataset_loader`')
