from dash import html, dcc
import time
import h5py
import plotly.graph_objects as go
import cv2


overlay = html.Div(
    id='overlay',
    style={'display': 'none'},
    children=[
        html.Div([
            html.Div(
                children = [
                    html.Div(
                    children = html.H2(
                        children = "Image Features Visualizations",
                        className="text-center h2"
                        
                    ), className="col-md-11"),
                    html.Div(
                    children = html.Button('Close', id='close-overlay', className = "btn btn-danger m-2",
                        n_clicks_timestamp=int(time.time() + 10)), className="col-md-1")
                ],
                className="row container-fluid bg-white"
            ),
            html.Div(
                id="inner-container",
                className="container-fluid",
                children=None
            )
        ])
    ])


def get_layout(title, is_3d=False):
    camera = dict(
        eye=dict(x=0, y=2, z=0)
    )

    return go.Layout(
        margin=dict(l=0, r=0, t=40, b=0),
        title=title,
        scene=dict(camera=camera if is_3d else None)
    )


def build_first_row(image, space, hist):

    img_disp = go.Figure(data=[go.Image(z=image)], layout=get_layout("Image"))

    red = space[:, 0]
    green = space[:, 2]
    blue = space[:, 2]
    c = space/255

    col_space = go.Figure(data=[go.Scatter3d(x=red, y=green, z=blue, mode="markers", marker=dict(
        size=1, color=c))], layout=get_layout("Color Space Visualization", True))

    col_hist = go.Figure(data=[go.Scatter(y=hist[:, 0], mode="lines", line=dict(color="red")),
                               go.Scatter(
                                   y=hist[:, 1], mode="lines", line=dict(color="green")),
                               go.Scatter(y=hist[:, 2], mode="lines", line=dict(color="blue"))],
                         layout=get_layout("color Histogram Visualization"))
    row = html.Div([
        html.Div(
            id='real-image',
            className='col-md-4',
            children=[
                dcc.Graph(
                    figure=img_disp
                )
            ]
        ),
        html.Div(
            id='col-space',
            className='col-md-4',
            children=[
                dcc.Graph(
                    figure=col_space
                )
            ]
        ),
        html.Div(
            id='col-hist',
            className='col-md-4',
            children=[
                dcc.Graph(
                    figure=col_hist
                )
            ]
        )
    ], className='row container-fluid mt-2')
    return row


def get_color_modes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    colormodes = {'hsv': hsv, 'yuv': yuv, 'ycrcb': ycrcb}
    modes = [go.Figure(data=[go.Image(z=img)], layout=get_layout(
        key)) for key, img in colormodes.items()]
    return modes

def build_second_row(row, image):

    modes = get_color_modes(image)
    images = []

    for mode in modes:
        img = html.Div(
            children=[
                dcc.Graph(
                    figure=mode
                )
            ],
            className="col-sm-4"
        )
        images.append(img)

    row = html.Div(
        className="row container-fluid mt-2",
        children=[
            html.Div(id="color-modes", children=[
                html.Div(
                    className="row",
                    children=images
                )
            ], className="col-md-9"),
            html.Div(id="metas", children=[
                html.H5("MetaData"),
                html.P("Height: {}px".format(row["Height"])),
                html.P("Width: {}px".format(row["Width"])),
                html.P("Color Channels: {}".format(row["Channel"])),
                html.P("Orientation: {}".format(row["Orientation"]))
            ], className="col-md-2 my-auto p-3 bg-white")
        ]
    )
    return row


def build_third_row(*features):
    title = ["Canny Edge Detection", "Entropy(smallest Kernel)", "Keypoints"]

    row = html.Div(
        className="row container-fluid mt-2",
        children=[
            html.Div(
            children=[
                dcc.Graph(
                    figure=go.Figure(data=[go.Heatmap(z=features[0], colorscale = "gray" )], layout=get_layout(
        title[0]))
                )
            ],
            className="col-sm-4"
        ),
        html.Div(
            children=[
                dcc.Graph(
                    figure=go.Figure(data=[go.Heatmap(z=features[1], colorscale = "gray" )], layout=get_layout(
        title[1]))
                )
            ],
            className="col-sm-4"
        ),
        html.Div(
            children=[
                dcc.Graph(
                    figure=go.Figure(data=[go.Image(z=features[2])], layout=get_layout(
        title[2]))
                )
            ],
            className="col-sm-4"
        )
        ]
    )
    return row

def build_visualization(row):

    h5path = row["Features_Fpath"]

    with h5py.File(h5path, "r") as f:
        image = f["Image"][:]
        col_space = f["ColorSpace"][:]
        col_hist = f["ColorHist"][:]
        edges = f["Edges"][:]
        entropy = f["Entropy"][:]
        keypoints = f["Keypoints"][:]

    first = build_first_row(image, col_space, col_hist)
    second = build_second_row(row, image)
    third = build_third_row(edges, entropy, keypoints)

    return [first, second, third]