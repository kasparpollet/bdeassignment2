import pandas as pd
import dask
from dask import dataframe as dd
import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

from scripts.database import DataBase
from scripts.clean import Clean
from scripts.keras import RNN, CNN


def display_wordcloud(reviews):
    """
    Create and display a wordclooud of a given series
    """
    Mask = np.array(Image.open('files/hotel.png'))
    image_colors = ImageColorGenerator(Mask)

    all_text = " ".join(review for review in reviews)
    wordcloud = WordCloud(background_color='white', mask=Mask).generate(all_text)
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_labeled_reviews(df):
    positive_reviews = (
        df[['Positive_Review']].copy()
        .rename({"Positive_Review": "Review"}, axis="columns")
    )

    negative_reviews = (
        df[['Negative_Review']].copy()
        .rename({"Negative_Review": "Review"}, axis="columns")
    )    

    positive_reviews['Label'] = 1
    negative_reviews['Label'] = 0

    # Combine the 2 dataframes  
    reviews = pd.concat([positive_reviews, negative_reviews])

    return reviews

def get_hotels_from_file():
    return pd.read_csv('files/Hotel_Reviews.csv')


if __name__ == "__main__":

    db = DataBase()
    df = db.get_all()
    df = get_labeled_reviews(df)
    df = Clean(df).df
    # print(df)

    # db.upload_data(df, name='labeled_reviews', collection=db.db['labeled_reviews'])
    # db.test(df)
    # print(df.columns)
    # display_wordcloud(df.Positive_Review)
    # display_wordcloud(df.Negative_Review)
    # import keras

    model = RNN(df, max_words=5000, batch_size=20, no_epochs=6, validation_split=0.2, verbosity=1, embedding_dims=128)
    model.create()