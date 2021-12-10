import pandas as pd
from dask import dataframe as dd

from pymongo import MongoClient


class DataBase:
    """
    The DataBase class creates a connections with a mongo database
    And holds methodes to interact with the database
    """
    
    def __init__(self, collection='reviews'):
        print('\nCreating database connection...')
        client = MongoClient("localhost:27017")
        self.db = client["assignment2"]
        self.collection = database["reviews"]

    def get_all(self, collection=self.collection):
        print('\nGetting data...')
        return pd.DataFrame(list(self.db[collection].find({})))

    def upload_data(self, df, name, collection=self.collection):
        """
        Upload a given pandas dataframe to the database wth a given table name
        """
        self.db[collection].insert_many(df.to_dict(name))
        print('\nSuccessful uploaded data')