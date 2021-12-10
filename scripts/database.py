import pandas as pd

from pymongo import MongoClient


class DataBase:
    """
    The DataBase class creates a connections with a mongo database
    And holds methodes to interact with the database
    """
    
    def __init__(self):
        print('\nCreating database connection...')
        client = MongoClient("localhost:27017")
        self.db = client.assignment2

    def get_all(self):
        print('Getting data...')
        return pd.DataFrame(list(self.db.reviews.find({})))

    def upload_data(self, df, name):
        """
        Upload a given pandas dataframe to the database wth a given table name
        """
        self.db.reviews.insert_many(df.to_dict(name))
        print('Successful uploaded data')