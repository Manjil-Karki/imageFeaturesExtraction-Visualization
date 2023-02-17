from dash import Dash, dcc, html, Input, Output, State
import time
import pandas as pd
from overlay import overlay, build_visualization
import plotly.graph_objs as go



app = Dash(external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"])

df = pd.read_csv("./visualize/dataframe.csv")
df.set_index("index", inplace=True)


def generate_thumbnail():
    paths = df["relative_path"]
    thumbnails = []
    for index, path in paths.items():
        thumbnail = html.A(
            id='open-overlay-'+index,
            n_clicks_timestamp=int(time.time()),
            children=[
                html.Img(
                    id="img-"+index,
                    src=path[12:]
                )
            ])
        thumbnails.append(thumbnail)
    return thumbnails


@app.callback(
    [
    Output(component_id='overlay', component_property='style'),
    Output(component_id='inner-container', component_property='children')
    ], 
    [Input(component_id='close-overlay', component_property='n_clicks_timestamp')]
        +[Input(component_id='open-overlay-{}'.format(_), component_property="n_clicks_timestamp") for _ in df.index],
        prevent_initial_call = True
)
def openClose_overlay(close, *clicks):
    if max(clicks) > close:
        ind = clicks.index(max(clicks))
        row = df.iloc[ind]
        visualization = build_visualization(row)
        return {'display': 'block'}, visualization
    else:
        return {'display': 'none'}, None 


if __name__ == "__main__":
    thumbnails = generate_thumbnail()

    app.layout = html.Div([
        html.H1(
            id= "heading",
            className= "h1 text-center",
            children="Acquired Images (click to see features)"
        ),
        html.Div(
            className="thumbnails",
            children=thumbnails
        ),
        dcc.Loading(
            id="overlay-loading",
            children = overlay,
            fullscreen = True,
            type="circle",
            className="bg-dark bg-opacity-50"
        )
    ])

    app.run_server()
