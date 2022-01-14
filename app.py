import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output
import pandas as pd
import numpy as np
import dash_table

from scripts.database import DataBase

db = DataBase()
# db.test()
number_of_reviews_per_hotel = db.get_amount_of_reviews_per_hotel()
avarage_score_per_hotel = db.get_avarage_score_per_hotel()
df = db.get_all()
df_old = df
top_hotels_to_view = 50

map_hover_text = []
hotels = df_old.Hotel_Name.unique()
for hotel in hotels:
    avr_score = avarage_score_per_hotel[hotel]
    number_of_reviews = number_of_reviews_per_hotel[hotel]
    map_hover_text.append([f'{hotel}, {avr_score} out of {number_of_reviews} reviews'])


df.drop_duplicates(subset=["Hotel_Name"],inplace=True)

app = dash.Dash()
app.title = 'Hotels'

app.layout = html.Div(
    html.Div([
        html.Div([
            html.H1(children='Hotels in Europes major cities'),

            html.Div(id='my-div'),
        ], className = 'row'),

        html.Div([
            html.Div(className = 'six columns', style={'display': 'inline-block'}, id='map'),
        ], className = 'row'),

        html.Div(
            children=[html.H4(children='Hotels'),
            dcc.Dropdown(
                id='select_hotel',
                options=[{'label': i, 'value': i} for i in df_old.Hotel_Name.unique()],
                multi=True, placeholder='Filter by Hotel'),
            html.Div(id='table-container')
        ]),

        html.Div([
            html.Div(className = 'six columns', id='number_of_reviews'),
            html.Div(className = 'six columns', id='avarage_score'),
            html.Div(className = 'six columns', id='score_amount'),
            html.Div(className = 'six columns', id='tags'),

            # html.Div([
            #     html.H4(children='Hotels in Europes major cities', style={'display': 'inline-block'}),
            # ], className = 'six columns'),

            # html.Div([
            #     dash_table.DataTable(
            #             id='table',
            #             columns=[{"name": i, "id": i} for i in df.columns],
            #             data=df.to_dict('records'),
            #         ),
            # ], className = 'six columns'),
        ], className = 'row')
    ])
)


@app.callback(
    Output("number_of_reviews", "children"), 
    Input("select_hotel", "value")
)
def update_number_of_reviews(hotels):
    number_of_reviews_per_hotel = db.get_amount_of_reviews_per_hotel(hotels)
    return dcc.Graph(
        figure={
            'data': [
                {'x': list(number_of_reviews_per_hotel.keys())[:top_hotels_to_view], 'y': list(number_of_reviews_per_hotel.values()), 'type': 'bar', 'name': 'Number of reviews'},
            ],
            'layout': {
                'title': f'Top {top_hotels_to_view} hotels with the most reviews' if len(number_of_reviews_per_hotel) >= 49 else f'Number of reviews of {len(number_of_reviews_per_hotel)} selected Hotels'
            }
        }
    )

@app.callback(
    Output("avarage_score", "children"), 
    Input("select_hotel", "value")
)
def update_avarage_score(hotels):
    avarage_score_per_hotel = db.get_avarage_score_per_hotel(hotels)
    return dcc.Graph(
        figure={
            'data': [
                {'x': list(avarage_score_per_hotel.keys())[:top_hotels_to_view], 'y': list(avarage_score_per_hotel.values()), 'type': 'bar', 'name': 'Score'},
            ],
            'layout': {
                'title': f'Top {top_hotels_to_view} hotels' if len(avarage_score_per_hotel) >= 49 else f'Avarage score of {len(avarage_score_per_hotel)} selected Hotels'
            }
        }
    )

@app.callback(
    Output("score_amount", "children"), 
    Input("select_hotel", "value")
)
def update_amount_of_scores(hotels):
    amount_of_scores = db.get_how_many_times_a_score_has_been_given(hotels)
    return dcc.Graph(
        figure={
            'data': [
                {'x': list(amount_of_scores.keys()), 'y': list(amount_of_scores.values()), 'type': 'bar', 'name': 'Amount'},
            ],
            'layout': {
                'title': f'How many times a score has been given'
            }
        }
    )

@app.callback(
    Output("tags", "children"), 
    Input("select_hotel", "value")
)
def update_tags(hotels):
    tags = db.test(hotels)
    return dcc.Graph(
        figure={
            'data': [
                {'x': list(tags.keys())[:top_hotels_to_view], 'y': list(tags.values()), 'type': 'bar', 'name': 'Amount'},
            ],
            'layout': {
                'title': f'Top {top_hotels_to_view} most used tags' if len(avarage_score_per_hotel) >= 49 else f'Most used tags of {len(avarage_score_per_hotel)} selected Hotels'
            }
        }
    )

@app.callback(
    Output("map", "children"), 
    Input("select_hotel", "value")
)
def update_map(hotels):

    if hotels:
        selected_hotels_df = df[df['Hotel_Name'].isin(hotels)]
        zoom = 16
    else: 
        selected_hotels_df = df
        zoom = 4
    
    layout_map = dict(
        autosize=True,
        height=600,
        width=1000,
        font=dict(color="#191A1A"),
        titlefont=dict(color="#191A1A", size='14'),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        hovermode="closest",
        plot_bgcolor='#fffcfc',
        paper_bgcolor='#fffcfc',
        legend=dict(font=dict(size=10), orientation='h'),
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lon = selected_hotels_df[selected_hotels_df['Hotel_Name']==hotels[-1]]['lng'].mean() if hotels else df['lng'].mean(),
                lat = selected_hotels_df[selected_hotels_df['Hotel_Name']==hotels[-1]]['lat'].mean() if hotels else df['lat'].mean()
            ),
            zoom=zoom,
        )
    )

    return dcc.Graph(
        id='MapPlot',
        figure={
            "data": [{
                "type": "scattermapbox",
                "lat": list(df.lat),
                "lon": list(df.lng),
                "hoverinfo": "text",
                "hovertext": map_hover_text,
                "mode": "markers",
                "name": list(df['Hotel_Name']),
                "marker": {
                    "size": 15,
                    "opacity": 0.7,
                    "color": '#222222',
                },
                "name":'Hotels'
            },
            {
                "type": "scattermapbox",
                "lat": list(selected_hotels_df.lat),
                "lon": list(selected_hotels_df.lng),
                "hoverinfo": "text",
                "hovertext": map_hover_text,
                "mode": "markers",
                "name": list(selected_hotels_df['Hotel_Name']),
                "marker": {
                    "size": 15,
                    "opacity": 1,
                    "color": '#F70F0F',
                },
                "name":'Selected Hotels'
            }],
            "layout": layout_map
        }                    
    ),

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css 414'
})

if __name__ == '__main__':
    app.run_server(debug=True)