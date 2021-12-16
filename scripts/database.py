import pandas as pd

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
        self.collection = self.db["reviews"]

    def get_all(self, collection=None):
        print('\nGetting data...')
        if not collection:
            collection = self.collection
        return pd.DataFrame(list(collection.find({})))

    def upload_data(self, df, name, collection=None):
        """
        Upload a given pandas dataframe to the database wth a given table name
        """
        if not collection:
            collection = self.collection
        collection.insert_many(df.to_dict(name))
        print('\nSuccessful uploaded data')