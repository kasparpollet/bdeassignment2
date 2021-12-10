import pandas as pd

from scripts.database import DataBase


def get_hotels_from_file():
    return pd.read_csv('files/Hotel_Reviews.csv')


if __name__ == "__main__":

    db = DataBase()
    df = db.get_all()
    print(df)