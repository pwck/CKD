import pandas as pd
import numpy as np
import logging

class LoadFiles():
    """ 
    Class for loading of files and data preparation

    Parameters: 
        None

    Attributes: 
        model (Dictionary): Dictionary of estimator objects to use for prediction
        df (Dictionary): Dictionary of Dataframes holding the data to be used for prediction
        target (String): Target to be predicted
        feats (Dictionary): Dictionary of features to use for prediction
        split_df (Dictionary): Dictionary of train test split Dataframe

    """
    
    def __init__(self): # Constructor
        # instance variable
        self.data_files = None
        self.alldata = dict()
        self.non_vitals = None
        self.vals_df = None
        self.drugs = None
        self.meds_df = None
        self.demo_df = None
        self.ckd_df = None
        self.ckd_train = None
        self.ckd_valid = None
        self.ckd_test = None
        
    def load_files(self):
        for file in self.data_files.keys():
            self.alldata[file] = pd.read_csv(self.data_files[file])
        self.drugs = set(self.alldata['meds']['drug'])
        return
    
    def check_file(self, file):
        """ 
        Helper function to check data 

        Parameters: 
            file (String): data file to be checked 

        Returns: 
            None

        """
        print(f'<<<--- info() for {file} --->>>')
        display(self.alldata[file].info())
        
        tmp = self.alldata[file]['id'].agg(['nunique','count','size'])
        print(f'\n<<<--- unique counts  for {file} --->>>')
        print(f'Count of distinct values\t\t: {tmp["nunique"]}')
        print(f'Count of only non-null values\t\t: {tmp["count"]}')
        print(f'Count of total values including null values\t: {tmp["size"]}')
        
        print(f'\n<<<--- describe() for {file} --->>>')
        display(self.alldata[file].describe())
        
        print(f'\n<<<--- head(10) for {file} --->>>')
        display(self.alldata[file].head(10))
        return
    
    def check_patient(self, id):
        """ 
        Function to check patient data 

        Parameters: 
            id (String): patient id to be checked 

        Returns: 
            None

        """
        for key in self.alldata.keys():
            df = self.alldata[key]
            print(f'Data for patient:{id} in file:{key}')
            display(df[df['id']==id])
            print(f'\n------------------------------')
        return
    
    def trendline(self, index, data, order=1):
        """ 
        Function to calculate trendline 

        Parameters: 
            index (List): List of index
            data (List): Data for trendline calculation
            order (Int): Degree of the fitting polynomial

        Returns: 
            Float

        """
        coeffs = np.polyfit(index, list(data), order)
        slope = coeffs[-2]
        return float(slope) 
    
    def get_values(self, id):
        """ 
        Function to get patient's value

        Parameters: 
            id (String): patient id's value to retrieve 

        Returns: 
            dict()

        """
        
        vals = dict()
        vals['id']=id
        for key in self.alldata.keys():
            if key in self.non_vitals:
                continue
            
            df = self.alldata[key]
            max_time = df[df["id"]==id]["time"].max()
            vals[key] = df[(df["id"]==id) & (df["time"]==max_time)]['value'].max()
            
            #Trend default set to 1, 1 if trend is going up 0 if trend is going down 
            vals[f'{key}_trend']=1
            if self.trendline(df[df["id"]==id]['time'], 
                df[df["id"]==id]['value']) < 0:
                vals[f'{key}_trend']=0
            
            logging.debug(f'Data for patient:{id} in file:{key} max:{max_time} val:{vals[key]} trend:{vals[f"{key}_trend"]}')
                
        return {id:vals}
        
    def merge_vals(self):
        """ 
        Function to merge all patients' value into one dataframe

        Parameters: 
            None 

        Returns: 
            None

        """
        
        val_main = dict()
        for id in self.alldata['stage']['id']:
            val_main.update(self.get_values(id))
        
        self.vals_df = pd.DataFrame(val_main).T
        return    
    
    def get_meds(self):
        """ 
        Function to merge all patients' meds taken into one dataframe

        Parameters: 
            None 

        Returns: 
            None

        """
        med_main = dict()
        med_id = set(self.alldata['meds']['id'])
        for id in self.alldata['stage']['id']:
            df = self.alldata['meds']
            df = df[df['id']==id]
            d_drugs = dict.fromkeys(self.drugs, 0)
            d_drugs['id'] = id
            if id in med_id:
                for d in set(df['drug']):
                    d_drugs[d] = 1
            med_main.update({id:d_drugs})
        
        self.meds_df = pd.DataFrame(med_main).T
        return
        
        
        
        