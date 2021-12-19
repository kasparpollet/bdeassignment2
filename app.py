import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import dash_table


df = pd.read_csv('files/Hotel_Reviews.csv')
df_old = df
# print(df_old)

print(df_old[df_old['Hotel_Name']=='Hotel Arena'])

extra = []
hotels = df_old.Hotel_Name.unique()
# for hotel in hotels:
#     reviews = df_old[df_old['Hotel_Name']==hotel]
#     avr_score = round(reviews.Reviewer_Score.mean(),2)
#     number_of_reviews = len(reviews)
#     extra.append([f'{hotel}, {avr_score} out of {number_of_reviews}'])


df.drop_duplicates(subset=["Hotel_Name"],inplace=True)

app = dash.Dash()
app.title = 'Hotels'

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
            lon = df['lng'].mean(),
            lat = df['lat'].mean()
        ),
        zoom=4,
    )
)

app.layout = html.Div(
    html.Div([
        html.Div([
            html.H1(children='Hotels in Europes major cities'),

            html.Div(id='my-div'),
        ], className = 'row'),

        # html.Br(),

        html.Div([
            html.Div([
                dcc.Graph(
                    id='MapPlot',
                    figure={
                        "data": [{
                            "type": "scattermapbox",
                            "lat": list(df.lat),
                            "lon": list(df.lng),
                            "hoverinfo": "text",
                            "hovertext": extra,
                            "mode": "markers",
                            "name": list(df['Hotel_Name']),
                            "marker": {
                                "size": 15,
                                "opacity": 0.7,
                                "color": '#F70F0F',
                            }
                        }],
                        "layout": layout_map
                    }                    
                ),
            ], className = 'six columns', style={'display': 'inline-block'}),

            html.Div([
                html.H4(children='Hotels in Europes major cities', style={'display': 'inline-block'}),
            ], className = 'six columns'),

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

if __name__ == '__main__':
    app.run_server(debug=True)