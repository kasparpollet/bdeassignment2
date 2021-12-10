import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import dash_table


df = pd.read_csv('files/Hotel_Reviews.csv')
df.drop_duplicates(subset=["lat"],inplace=True)

# Subset dataframe to show some specific columns in dash web app
df1 = df[['Hotel_Name', 'lat', 'lng']]

# Find Lat Long center
lat_center = sum(df['lat'])/len(df['lat'])
long_center = sum(df['lng'])/len(df['lng'])

# Find Lat Long center
lat_center = sum(df['lat'])/len(df['lat'])
long_center = sum(df['lng'])/len(df['lng'])

app = dash.Dash()
app.title = 'Open Street Map'

layout_map = dict(
    autosize=True,
    height=500,
    weidth=100,
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
            lon = long_center,
            lat = lat_center
        ),
        zoom=2,
    )
)

app.layout = html.Div(
    html.Div([
        html.Div([
            html.H1(children='Hotels in Eurpoes major cities'),

            html.Div(id='my-div'),
        ], className = 'row'),

        html.Br(),

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
                            "hovertext": list(df['Hotel_Name']),
                            "mode": "markers",
                            "name": list(df['Hotel_Name']),
                            "marker": {
                                "size": 15,
                                "opacity": 0.7,
                                "color": '#F70F0F'
                            }
                        }],
                        "layout": layout_map
                    }
                ),
            ], className = 'six columns'),

            html.Div([
                dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in df1.columns],
                        data=df1.to_dict('records'),
                    ),
            ], className = 'six columns'),
        ], className = 'row')
    ])
)

if __name__ == '__main__':
    app.run_server(debug=True)