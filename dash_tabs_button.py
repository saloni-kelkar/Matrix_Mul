import dash
import json
import numpy as np
import pandas as pd
import networkx as nx
import dash_bio as dashbio
import plotly.express as px
from math import floor, ceil
import plotly.graph_objects as go
from warnings import simplefilter
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from coclust.coclustering import CoclustMod
from dash import Dash, dcc, html, Input, Output, callback, State

simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None

full_mcm = pd.read_excel('Data/Whisky_Full.xlsx', usecols='A:IP').to_numpy()

solutes_df = pd.read_excel('Data/MCM_whisky.xlsx', skiprows=2, sheet_name='Features', usecols='A:E', nrows=240, header=None,
                                  names=['I', 'U1', 'U2', 'U3', 'U4'], dtype={'I': np.float64, 'U1': np.float64, 'U2': np.float64, 'U3': np.float64, 'U4': np.float64})
solutes_df['in_range'] = False

solvents_df = pd.read_excel('Data/MCM_whisky.xlsx', skiprows=2, sheet_name='Features', usecols='H:L', nrows=250, header=None,
                                   names=['J', 'V1', 'V2', 'V3', 'V4'], dtype={'J': np.float64, 'V1': np.float64, 'V2': np.float64, 'V3': np.float64, 'V4': np.float64})
solvents_df['in_range'] = False

solutes_descriptors_df = pd.read_excel('Data/Descriptors.xlsx', skiprows=1, sheet_name='Solutes', usecols=[1, 2, 6, 7, 8], header=None,
                                       names=['I', 'Component Name', 'Molecular Formula', 'CAS Number', 'UNIFAC Groups'])
solutes_df = pd.merge(solutes_df, solutes_descriptors_df, on='I', how='outer')

solvents_descriptors_df = pd.read_excel('Data/Descriptors.xlsx', skiprows=1, sheet_name='Solvents', usecols=[1, 2, 6, 7, 8], header=None,
                                       names=['J', 'Component Name', 'Molecular Formula', 'CAS Number', 'UNIFAC Groups'])
solvents_df = pd.merge(solvents_df, solvents_descriptors_df, on='J', how='outer')

densities_df = pd.read_excel('density_columns.xlsx', usecols='A:D', skiprows=1, nrows=60000, header=None,
                             names=['first_density', 'second_density', 'third_density', 'fourth_density'])

combined_array = np.column_stack((np.outer(solutes_df['U1'], solvents_df['V1']).ravel(), np.outer(solutes_df['U2'], solvents_df['V2']).ravel(),
                         np.outer(solutes_df['U3'], solvents_df['V3']).ravel(), np.outer(solutes_df['U4'], solvents_df['V4']).ravel(), full_mcm.ravel()))

features_df = pd.DataFrame(combined_array, columns=['first', 'second', 'third', 'fourth', 'mcm'])
features_df['first_density'] = densities_df['first_density'].values
features_df['second_density'] = densities_df['second_density'].values
features_df['third_density'] = densities_df['third_density'].values
features_df['fourth_density'] = densities_df['fourth_density'].values
features_df['in_range'] = False

def factors(n):
    factors_set = set()
    for i in range(3, int(n ** 0.5) + 1):
        if n % i == 0:
            factors_set.add(i)
    return factors_set

def calculate_number_of_clusters_in_network(filtered_data_matrix):
    wcss = {}

    for i in range(2, 6):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(filtered_data_matrix)
        wcss[i] = (kmeans.inertia_)
    greatest_difference = None
    item_with_greatest_difference = None
    sorted_keys = sorted(wcss.keys())
    for i in range(1, len(sorted_keys)):
        current_item = sorted_keys[i]
        prev_item = sorted_keys[i - 1]
        current_value = wcss[current_item]
        prev_value = wcss[prev_item]
        difference = prev_value - current_value
        if greatest_difference is None or difference > greatest_difference:
            greatest_difference = difference
            item_with_greatest_difference = current_item

    return item_with_greatest_difference

def calculate_network_node_values(feature_col, u_col, v_col, negative_value, positive_value):
    filtered_data_matrix = features_df[feature_col][((negative_value[0] <= features_df[feature_col]) & (features_df[feature_col] <= negative_value[1])) |
    ((positive_value[0] <= features_df[feature_col]) & (features_df[feature_col] <= positive_value[1]))].to_numpy()
    if len(factors(len(filtered_data_matrix))) == 0:
        divisor = sorted(factors((len(filtered_data_matrix) + 1))).pop()
        filtered_data_matrix=np.append(filtered_data_matrix, 0)
    else:
        divisor = sorted(factors(len(filtered_data_matrix))).pop()
    filtered_data_matrix = np.reshape(filtered_data_matrix, (len(filtered_data_matrix) // divisor, divisor))
    number_of_clusters = calculate_number_of_clusters_in_network(filtered_data_matrix)
    model = CoclustMod(number_of_clusters)
    model.fit(filtered_data_matrix)
    row_labels = model.row_labels_
    column_labels = model.column_labels_

    from_list, to_list = [], []
    median_from_list, median_to_list = [], []
    from_list_values, to_list_values = [], []
    for target_cluster in range(0, number_of_clusters):
        rows_in_cluster = np.array([index for index, element in enumerate(row_labels) if element == target_cluster])
        columns_in_cluster = np.array([index for index, element in enumerate(column_labels) if element == target_cluster])
        values_in_cluster = filtered_data_matrix[rows_in_cluster[:, np.newaxis].astype(int), columns_in_cluster.astype(int)]
        filtered_values = features_df.loc[features_df[feature_col].isin(values_in_cluster.ravel()), feature_col]

        possible_u_values = np.array(filtered_values)[:, np.newaxis] / solvents_df[v_col].to_numpy()
        from_list_values = solutes_df.loc[solutes_df[u_col].isin(possible_u_values.ravel()), u_col].to_numpy()

        possible_v_values = np.array(filtered_values)[:, np.newaxis] / solutes_df[u_col].to_numpy()
        to_list_values = solvents_df.loc[solvents_df[v_col].isin(possible_v_values.ravel()), v_col].to_numpy()

        median_from_list.append(np.median(from_list_values))
        median_to_list.append(np.median(to_list_values))

        from_list.append(from_list_values)
        to_list.append(to_list_values)
    return from_list, to_list, median_from_list, median_to_list

def generate_network_graph(from_list, to_list, median_from_list, median_to_list):
    G = nx.Graph()
    for x in range(len(median_from_list)):
        for y in range(len(median_to_list)):
            G.add_edge(median_from_list[x], median_to_list[y], weight=(len(from_list[x]) + len(to_list[y])))
    G.add_nodes_from(median_from_list, bipartite=0, label='Solute i')
    G.add_nodes_from(median_to_list, bipartite=1, label='Solvent j')

    pos1 = nx.bipartite_layout(G, median_from_list, scale=10)
    for n, p in pos1.items():
        G.nodes[n]['pos'] = p

    return G

def plot_network_graph(G, negative_slider_value, positive_slider_value):
    edge_x1 = []
    edge_y1 = []
    edge_widths = []
    for edge in G.edges(data=True):
        from_node, to_node, weight = edge
        x0, y0 = G.nodes[from_node]['pos']
        x1, y1 = G.nodes[to_node]['pos']
        edge_x1.extend([x0, x1, None])
        edge_y1.extend([y0, y1, None])
        edge_widths.append(weight['weight'])

    min_width = 0.1
    max_width = 1.0
    normalized_widths = np.interp(edge_widths, (min(edge_widths), max(edge_widths)), (min_width, max_width))

    edge_traces = []
    for i, w in enumerate(normalized_widths):
        edge_trace = go.Scatter(
            x=edge_x1[i * 3:i * 3 + 3],
            y=edge_y1[i * 3:i * 3 + 3],
            mode='lines',
            line=dict(width=w, color='black'),
            hoverinfo='none',
        )
        edge_traces.append(edge_trace)

    solute_color = 'Green'
    solvent_color = 'Purple'

    node_x1 = []
    node_y1 = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x1.append(x)
        node_y1.append(y)
    node_trace = go.Scatter(x=node_x1, y=node_y1, text=[], mode='markers+text', hoverinfo='text', textposition='top center', texttemplate='%{text:.4f}',
                            marker=dict(showscale=False, size=12, line=dict(width=0),
                                        color=[solute_color if G.nodes[node]['bipartite'] == 0 else solvent_color for node in G.nodes()]))

    for node, adjacencies in enumerate(G.adjacency()):
        node_info = adjacencies[0]
        node_trace['text'] += tuple([node_info])


    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=21, l=5, r=5, t=40),
                        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False, mirror=True),
                        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False, mirror=True)))

    fig.update_layout(title=f"Solute and solvent values within [{negative_slider_value[0]:.2f}, {negative_slider_value[1]:.2f}] "
                            f"and [{positive_slider_value[0]:.2f}, {positive_slider_value[1]:.2f}]", height=1000)

    return fig

def plot_network_dist_data(from_list, to_list, median_from_list, median_to_list, selected_node, u_col, v_col):
    selected_node = np.float64(selected_node)
    if selected_node in median_from_list:
        index_matching_node = median_from_list.index(selected_node)
        cluster_values = from_list[index_matching_node]
        cluster_names = solutes_df.loc[solutes_df[u_col].isin(cluster_values)]
        columns_to_display = ['I', 'Component Name', 'Molecular Formula', 'CAS Number', 'UNIFAC Groups']
    elif selected_node in median_to_list:
        index_matching_node = median_to_list.index(selected_node)
        cluster_values = to_list[index_matching_node]
        cluster_names = solvents_df.loc[solvents_df[v_col].isin(cluster_values)]
        columns_to_display = ['J', 'Component Name', 'Molecular Formula', 'CAS Number', 'UNIFAC Groups']
    else:
        return False, None, None
    fig1 = go.Figure(data=go.Violin(y=cluster_values, box_visible=True, meanline_visible=True, name='Distribution of the cluster', points='outliers'))

    fig2 = go.Figure(data=go.Table(header=dict(values=list(columns_to_display), align='left', line_color='darkslategray', height=35),
                                   columnwidth=[4, 15, 10, 10, 10],
                                   cells=dict(values=[cluster_names[col_name] for col_name in columns_to_display],
                                              line_color='darkslategray', fill_color='lightcyan',
                                              align='left', font_size=12, height=30)))

    return True, fig1, fig2

def plot_scatterplot(df, feature_col, feature_density_col, x_label):
    fig = px.scatter(x=df[feature_col], y=df['mcm'], color=df[feature_density_col],
                     trendline='ols', color_continuous_scale='Plasma',
                     labels={'x': x_label, 'y': 'ln(gamma_ij)^MCM'},  trendline_color_override="Black")
    fig.update_traces(hovertemplate='x: %{x}<br>y: %{y}')
    fig.update_layout(coloraxis_colorbar=dict(title='Density'))
    return fig

def plot_updated_scatterplot(df, feature_col, feature_density_col, x_label):
    fig = px.scatter(x=df[feature_col], y=df['mcm'], color=df[feature_density_col],
                     trendline='ols', color_continuous_scale='Plasma',
                     labels={'x': x_label, 'y': 'ln(gamma_ij)^MCM'},  trendline_color_override="Black")
    x_min_plot = np.min(df[feature_col])
    x_max_plot = np.max(df[feature_col])
    y_min_plot = np.min(df['mcm'])
    y_max_plot = np.max(df['mcm'])
    x_min_plot_coord = x_min_plot - ((x_max_plot - x_min_plot) * 0.1)
    x_max_plot_coord = x_max_plot + ((x_max_plot - x_min_plot) * 0.1)
    y_min_plot_coord = y_min_plot - ((y_max_plot - y_min_plot) * 0.1)
    y_max_plot_coord = y_max_plot + ((y_max_plot - y_min_plot) * 0.1)
    x_min_in_range = np.min(df.loc[df['in_range'] == True, feature_col])
    x_max_in_range = np.max(df.loc[df['in_range'] == True, feature_col])
    fig.update_layout(
        shapes=[
            dict(
                type='rect',
                x0=x_min_plot_coord, y0=y_min_plot_coord,
                x1=x_min_in_range, y1=y_max_plot_coord,
                fillcolor='rgba(128, 128, 128, 0.8)',
                line=dict(color='rgba(0, 0, 0, 0)'),
                layer='above',
            ),
            dict(
                type='rect',
                x0=x_max_in_range, y0=y_min_plot_coord,
                x1=x_max_plot_coord, y1=y_max_plot_coord,
                fillcolor='rgba(128, 128, 128, 0.8)',
                line=dict(color='rgba(0, 0, 0, 0)'),
                layer='above',
            )
        ]
    )
    fig.update_layout(coloraxis_colorbar=dict(title='Density'))

    return fig

def plot_histogram(df, col, plot_title, axis_title):
    fig = px.histogram(df, x=df[col], nbins=30, marginal='rug', color_discrete_sequence=['DarkBlue'])
    fig.update_layout(title_text=plot_title, xaxis_title_text=axis_title, yaxis_title_text='Count')
    return fig

def plot_updated_histogram(df, col, plot_title, axis_title):
    fig = px.histogram(df, x=df[col], color=df['in_range'], color_discrete_map={True: 'Tomato', False: 'DarkBlue'}, nbins=30, marginal='rug')
    fig.update_layout(title_text=plot_title, xaxis_title_text=axis_title, yaxis_title_text='Count', showlegend=False)
    return fig

def calculate_in_range_solute_solvent_values(in_range_product, u_col, v_col):
    possible_u_values = np.array(in_range_product)[:, np.newaxis] / solvents_df[v_col].to_numpy()
    condition_u_values = solutes_df[u_col].isin(possible_u_values.flatten())
    solutes_df.loc[condition_u_values, 'in_range'] = True

    possible_v_values = np.array(in_range_product)[:, np.newaxis] / solutes_df[u_col].to_numpy()
    condition_v_values = solvents_df[v_col].isin(possible_v_values.flatten())
    solvents_df.loc[condition_v_values, 'in_range'] = True
    return solutes_df, solvents_df

def convert_to_categorical(float_list):
    lower_bound = -8
    upper_bound = 4
    categories = {}
    for i in range(int(lower_bound / 0.60), int(upper_bound / 0.60) + 1):
        lower_interval = i * 0.60
        upper_interval = (i + 1) * 0.60
        category = f"{lower_interval:.2f} to {upper_interval:.2f}"
        categories[(lower_interval, upper_interval)] = category

    categorical_values = []
    for value in float_list:
        found_category = False
        for (lower_interval, upper_interval), category in categories.items():
            if lower_interval <= value <= upper_interval:
                categorical_values.append(category)
                found_category = True
                break
        if not found_category:
            if value < lower_bound:
                categorical_values.append(f"less than {lower_bound}")
            elif value > upper_bound:
                categorical_values.append(f"greater than {upper_bound}")
    return categorical_values

def display_solute_violin_plots():
    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(go.Violin(y=solutes_df['U1'], name='U1 Distribution'), row=1, col=1)
    fig.add_trace(go.Violin(y=solutes_df['U2'], name='U2 Distribution'), row=1, col=2)
    fig.add_trace(go.Violin(y=solutes_df['U3'], name='U3 Distribution'), row=2, col=1)
    fig.add_trace(go.Violin(y=solutes_df['U4'], name='U4 Distribution'), row=2, col=2)
    fig.update_traces(box_visible=True, meanline_visible=True, points='all', fillcolor='DodgerBlue')
    fig.update_layout(showlegend=False, height=600)

    return fig

def display_solvent_violin_plots():
    fig = make_subplots(rows=2, cols=2)
    fig.add_trace(go.Violin(y=solvents_df['V1'], name='V1 Distribution'), row=1, col=1)
    fig.add_trace(go.Violin(y=solvents_df['V2'], name='V2 Distribution'), row=1, col=2)
    fig.add_trace(go.Violin(y=solvents_df['V3'], name='V3 Distribution'), row=2, col=1)
    fig.add_trace(go.Violin(y=solvents_df['V4'], name='V4 Distribution'), row=2, col=2)
    fig.update_traces(box_visible=True, meanline_visible=True, points='all', fillcolor='DodgerBlue')
    fig.update_layout(showlegend=False, height=600)
    return fig

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='solutes-solvents-tab', children=[
        dcc.Tab(label='Solutes and solvents', value='solutes-solvents-tab',
                style={
                    'fontFamily': 'Helvetica',
                    'fontSize': '20px',
                    'border': '1px solid #ccc',
                    'cursor': 'pointer',
                },
                children=[
                    html.Div([
                        dcc.Store(id="data-store"),
                        dcc.Loading(
                            id="solutes-solvents-loading",
                            children=[
                                html.Div(
                                    [
                                        dbc.RadioItems(
                                            id="solutes-solvents-options-config",
                                            class_name="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[
                                                {"label": "U1*V1", "value": 'first'},
                                                {"label": "U2*V2", "value": 'second'},
                                                {"label": "U3*V3", "value": 'third'},
                                                {"label": "U4*V4", "value": 'fourth'},
                                            ],
                                            value='first',
                                        ),
                                    ],
                                    className="radio-group", style={"marginTop": "40px"},
                                ),
                                dcc.Graph(id="scatter-plot"),
                                html.Div([
                                    dcc.RangeSlider(id='negative-network-range-slider', max=-1,
                                                    allowCross=False, step=0.5,
                                                    tooltip={"placement": "bottom", "always_visible": False})
                                ], style={"width": "45%", "display": "inline-block", "margin-left": "40px"}),
                                html.Div([
                                    dcc.RangeSlider(id='positive-network-range-slider', min=1,
                                                    allowCross=False, step=0.5,
                                                    tooltip={"placement": "bottom", "always_visible": False})
                                ], style={"width": "45%", "display": "inline-block", "align": "center", "marginLeft": "20px"}),
                                html.Div([
                                    html.Div([
                                        dcc.Graph(id="histo-gram1"), dcc.Graph(id="histo-gram2"),
                                    ], style={"width": "50%", "display": "inline-block"}),
                                    html.Div([
                                        dcc.Graph(id="network-plot", config={"editable": False, 'scrollZoom': False, 'displayModeBar': False})
                                    ],style={"width": "50%", 'display': "inline-block"})
                                ], style={"width": "100%", 'display': "inline-block"}),
                            ], type="circle"),
                        dbc.Modal(
                            id="network_dist_modal",
                            size='lg',
                            is_open=False,
                            backdrop=True,
                            keyboard=True,
                            children=[dbc.ModalBody(
                                html.Div([dcc.Graph(id='network_modal_dist_graph',
                                                    style={'width': '350px', 'height': '400px',  'margin': '0%', 'padding': '0%', 'display': 'inline-block'}),
                                          dcc.Graph(id='network_modal_dist_table',
                                                    style={'width': '500px', 'height': '400px', 'margin': '0%', 'padding': '0%', 'display': 'inline-block'})
                                          ], style={'alignItems': 'center', "display": 'flex'})
                            ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="network_close_modal_button", className="ms-auto")),
                            ],
                            style={'position': 'fixed', 'top': '50%', 'left': '40%', 'transform': 'translate(-50%, -50%)', 'zIndex': '2000'},
                        )
                    ]),
                ]),
        dcc.Tab(label='Solutes', value='solutes-tab',
                style={
                    'fontFamily': 'Helvetica',
                    'fontSize': '20px',
                    'border': '1px solid #ccc',
                    'cursor': 'pointer',
                },
                children=[
                    html.Div([
                        dcc.Loading(
                            id="solutes-loading",
                            children=[
                                dcc.Graph(id="solute-parallel-coords-plot", style={"margin": 0, "padding": 0}),
                                html.Div([
                                    dbc.RadioItems(
                                        id="solute-options-config",
                                        class_name="btn-group",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[
                                            {"label": "U1", "value": 'first'},
                                            {"label": "U2", "value": 'second'},
                                            {"label": "U3", "value": 'third'},
                                            {"label": "U4", "value": 'fourth'},
                                        ],
                                        value='first',
                                    ),
                                ],
                                    className="radio-group"
                                ),
                                dcc.Graph(id="solute-clustergram", style={"margin": 0, "padding": 0})
                            ], type='circle'),
                    ]),
                ]),
        dcc.Tab(label='Solvents', value='solvents-tab',
                style={
                    'fontFamily': 'Helvetica',
                    'fontSize': '20px',
                    'border': '1px solid #ccc',
                    'cursor': 'pointer',
                },
                children=[
                    html.Div([
                        dcc.Loading(
                            id="solvents-loading",
                            children=[
                                dcc.Graph(id="solvent-parallel-coords-plot", style={"margin": 0, "padding": 0}),
                                html.Div([
                                    dbc.RadioItems(
                                            id="solvent-options-config",
                                            class_name="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[
                                                {"label": "V1", "value": 'first'},
                                                {"label": "V2", "value": 'second'},
                                                {"label": "V3", "value": 'third'},
                                                {"label": "V4", "value": 'fourth'},
                                            ],
                                            value='first',
                                        ),
                                ],
                                    className="radio-group"
                                ),
                                dcc.Graph(id="solvent-clustergram", style={"margin": 0, "padding": 0})
                            ], type='circle')
                    ]),
                ]),
        dcc.Tab(label='Overview', value='correlations-tab',
                style={
                    'fontFamily': 'Helvetica',
                    'fontSize': '20px',
                    'border': '1px solid #ccc',
                    'cursor': 'pointer',
                },
                children=[
                    html.Div([
                        dcc.Loading(
                            id="correlations-loading",
                            children=[
                                dcc.Graph(id="correlation-matrix", style={"margin": 0, "padding": 0}),
                                html.Div(
                                    [
                                        dbc.RadioItems(
                                            id="correlations-options-config",
                                            class_name="btn-group",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary",
                                            labelCheckedClassName="active",
                                            options=[
                                                {"label": "Solutes", "value": 'solutes'},
                                                {"label": "Solvents", "value": 'solvents'},
                                            ],
                                            value='solutes',
                                        ),
                                    ],
                                    className="radio-group",
                                ),
                                dcc.Graph(id="violin-plots", style={"margin": 0, "padding": 0}),
                            ], type='circle'),
                        dbc.Modal(
                            id="correlations_det_modal",
                            size='sm',
                            is_open=False,
                            backdrop=True,
                            keyboard=True,
                            children=[dbc.ModalBody(
                                    dcc.Graph(id='correlations_modal_det_table'),
                                    style={"padding": "0px", "width": "600px"}
                                ),
                                dbc.ModalFooter(
                                    dbc.Button("Close", id="correlations_close_modal_button", className="ms-auto"),
                                    style={"padding": "0px", "width": "600px"}
                                ),
                            ],
                            style={'position': 'fixed', 'top': '50%', 'left': '40%', 'overflow-y': 'hidden',
                                   'transform': 'translate(-50%, -50%)', 'zIndex': '2000'},

                        )
                    ]),
                ])

    ]),
])

@app.callback(
    [Output("scatter-plot", "figure"), Output("histo-gram1", "figure"), Output("histo-gram2", "figure"), Output("network-plot", "figure"),
     Output("data-store", "data"), Output("negative-network-range-slider", "min"), Output("positive-network-range-slider", "max"),
     Output("negative-network-range-slider", "value"), Output("positive-network-range-slider", "value"),
     Output("negative-network-range-slider", "marks"), Output("positive-network-range-slider", "marks")],
    [Input("solutes-solvents-options-config", "value")])
def display_solute_solvent_charts(option):
    if option == 'first':
        negative_slider_min = floor(np.min(features_df['first']))
        positive_slider_max = ceil(np.max(features_df['first']))
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('first', 'U1', 'V1', [negative_slider_min, -1], [1, positive_slider_max])
        fig1 = plot_scatterplot(features_df, 'first', 'first_density', 'U1*V1')
        fig2 = plot_histogram(solutes_df, 'U1', 'U1 Distribution', 'U1')
        fig3 = plot_histogram(solvents_df, 'V1', 'V1 Distribution', 'V1')
    elif option == 'second':
        negative_slider_min = floor(np.min(features_df['second']))
        positive_slider_max = ceil(np.max(features_df['second']))
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('second', 'U2', 'V2', [negative_slider_min, -1], [1, positive_slider_max])
        fig1 = plot_scatterplot(features_df, 'second', 'second_density', 'U2*V2')
        fig2 = plot_histogram(solutes_df, 'U2', 'U2 Distribution', 'U2')
        fig3 = plot_histogram(solvents_df, 'V2', 'V2 Distribution', 'V2')
    elif option == 'third':
        negative_slider_min = floor(np.min(features_df['third']))
        positive_slider_max = ceil(np.max(features_df['third']))
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('third', 'U3', 'V3', [negative_slider_min, -1], [1, positive_slider_max])
        fig1 = plot_scatterplot(features_df, 'third', 'third_density', 'U3*V3')
        fig2 = plot_histogram(solutes_df, 'U3', 'U3 Distribution', 'U3')
        fig3 = plot_histogram(solvents_df, 'V3', 'V3 Distribution', 'V3')
    elif option == 'fourth':
        negative_slider_min = floor(np.min(features_df['fourth']))
        positive_slider_max = ceil(np.max(features_df['fourth']))
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('third', 'U4', 'V4', [negative_slider_min, -1], [1, positive_slider_max])
        fig1 = plot_scatterplot(features_df, 'fourth', 'fourth_density', 'U4*V4')
        fig2 = plot_histogram(solutes_df, 'U4', 'U4 Distribution', 'U4')
        fig3 = plot_histogram(solvents_df, 'V4', 'V4 Distribution', 'V4')

    negative_slider_value = [negative_slider_min, -1]
    positive_slider_value = [1, positive_slider_max]
    negative_slider_marks = {"{:.2f}".format(value): {"label": "{:.2f}".format(value)} for value in np.linspace(negative_slider_min, -1, num=10)}
    positive_slider_marks = {"{:.2f}".format(value): {"label": "{:.2f}".format(value)} for value in np.linspace(1, positive_slider_max, num=10)}

    G = generate_network_graph(from_list, to_list, median_from_list, median_to_list)
    fig4 = plot_network_graph(G, negative_slider_value, positive_slider_value)

    from_list_storage = [array.tolist() for array in from_list]
    to_list_storage = [array.tolist() for array in to_list]
    median_from_list_storage = [array.tolist() for array in median_from_list]
    median_to_list_storage = [array.tolist() for array in median_to_list]
    storage_data = json.dumps([from_list_storage, to_list_storage, median_from_list_storage, median_to_list_storage])
    return fig1, fig2, fig3, fig4, storage_data, negative_slider_min, positive_slider_max, negative_slider_value, positive_slider_value, negative_slider_marks, positive_slider_marks

@app.callback(
    Output("scatter-plot", "figure", allow_duplicate=True),
    [Input("histo-gram1", "clickData"), Input("histo-gram2", "clickData"),
     Input("scatter-plot", "relayoutData"), Input("solutes-solvents-options-config", "value")],
    prevent_initial_call=True)
def update_solute_solvent_scatterplot(uClickData, vClickData, relayoutData, option):
    ctx = dash.callback_context
    triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]
    print(triggered_component)
    if triggered_component == 'solutes-solvents-options-config':
        return dash.no_update
    if triggered_component == 'scatter-plot' and uClickData is None and vClickData is None:
        return dash.no_update

    if triggered_component == 'histo-gram1' and uClickData is not None:
        for value in uClickData.values():
            in_range_u_indices = np.array(value[0].get('pointNumbers'))
            if option == 'first':
                in_range_u_values = [solutes_df['U1'][i] for i in in_range_u_indices]
                in_range_product_values = np.outer(in_range_u_values, solvents_df['V1'])
                condition_product_values = features_df['first'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'first', 'first_density', 'U1*V1')
            if option == 'second':
                in_range_u_values = [solutes_df['U2'][i] for i in in_range_u_indices]
                in_range_product_values = np.outer(in_range_u_values, solvents_df['V2'])
                condition_product_values = features_df['second'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'second', 'second_density', 'U2*V2')
            elif option == 'third':
                in_range_u_values = [solutes_df['U3'][i] for i in in_range_u_indices]
                in_range_product_values = np.outer(in_range_u_values, solvents_df['V3'])
                condition_product_values = features_df['third'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'third', 'third_density', 'U3*V3')
            elif option == 'fourth':
                in_range_u_values = [solutes_df['U4'][i] for i in in_range_u_indices]
                in_range_product_values = np.outer(in_range_u_values, solvents_df['V4'])
                condition_product_values = features_df['fourth'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'fourth', 'fourth_density', 'U4*V4')
        return fig

    elif triggered_component == 'histo-gram2' and vClickData is not None:
        for value in vClickData.values():
            in_range_v_indices = np.array(value[0].get('pointNumbers'))
            if option == 'first':
                in_range_v_values = [solvents_df['V1'][i] for i in in_range_v_indices]
                in_range_product_values = np.outer(in_range_v_values, solutes_df['U1'])
                condition_product_values = features_df['first'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'first', 'first_density', 'U1*V1')
            if option == 'second':
                in_range_v_values = [solvents_df['V2'][i] for i in in_range_v_indices]
                in_range_product_values = np.outer(in_range_v_values, solvents_df['U2'])
                condition_product_values = features_df['second'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'second', 'second_density', 'U2*V2')
            elif option == 'third':
                in_range_v_values = [solvents_df['V3'][i] for i in in_range_v_indices]
                in_range_product_values = np.outer(in_range_v_values, solvents_df['U3'])
                condition_product_values = features_df['third'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'third', 'third_density', 'U3*V3')
            elif option == 'fourth':
                in_range_v_values = [solvents_df['V4'][i] for i in in_range_v_indices]
                in_range_product_values = np.outer(in_range_v_values, solvents_df['U4'])
                condition_product_values = features_df['fourth'].isin(in_range_product_values.flatten())
                features_df.loc[condition_product_values, 'in_range'] = True
                fig = plot_updated_scatterplot(features_df, 'fourth', 'fourth_density', 'U4*V4')

    elif (triggered_component == 'scatter-plot') and (uClickData is not None or vClickData is not None) and (relayoutData is not None):
        print('triggered')
        if option == 'first':
            fig = plot_scatterplot(features_df, 'first', 'first_density', 'U1*V1')
        if option == 'second':
            fig = plot_scatterplot(features_df, 'second', 'second_density', 'U2*V2')
        elif option == 'third':
            fig = plot_updated_scatterplot(features_df, 'third', 'third_density', 'U3*V3')
        elif option == 'fourth':
            fig = plot_scatterplot(features_df, 'fourth', 'fourth_density', 'U4*V4')
    features_df['in_range'] = False

    return fig

@app.callback(
    [Output("histo-gram1", "figure", allow_duplicate=True), Output("histo-gram2", "figure", allow_duplicate=True)],
    [Input("solutes-solvents-options-config", "value"), Input("scatter-plot", "relayoutData"),
     Input("negative-network-range-slider", "value"), Input("positive-network-range-slider", "value")],
    prevent_initial_call=True)
def update_solute_solvent_histograms(option, relayoutData, negative_slider_value, positive_slider_value):
    ctx = dash.callback_context
    triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_component == 'drop-down':
        return [dash.no_update, dash.no_update]

    if triggered_component == 'scatter-plot':
        if "xaxis.autorange" in relayoutData:
            if option == 'first':
                fig2 = plot_histogram(solutes_df, 'U1', 'U1 Distribution', 'U1')
                fig3 = plot_histogram(solvents_df, 'V1', 'V1 Distribution', 'V1')
            elif option == 'second':
                fig2 = plot_histogram(solutes_df, 'U2', 'U2 Distribution', 'U2')
                fig3 = plot_histogram(solvents_df, 'V2', 'V2 Distribution', 'V2')
            elif option == 'third':
                fig2 = plot_histogram(solutes_df, 'U3', 'U3 Distribution', 'U3')
                fig3 = plot_histogram(solvents_df, 'V3', 'V3 Distribution', 'V3')
            else:
                fig2 = plot_histogram(solutes_df, 'U4', 'U4 Distribution', 'U4')
                fig3 = plot_histogram(solvents_df, 'V4', 'V4 Distribution', 'V4')
            return fig2, fig3

        elif "xaxis.range[0]" not in relayoutData:
            return [dash.no_update, dash.no_update]

        else:
            for key in relayoutData.keys():
                if (key == 'xaxis.range[0]'):
                    min_x = relayoutData[key]
                if (key == 'xaxis.range[1]'):
                    max_x = relayoutData[key]
                if (key == 'yaxis.range[0]'):
                    min_y = relayoutData[key]
                if (key == 'yaxis.range[1]'):
                    max_y = relayoutData[key]

            if option == 'first':
                in_range_product = features_df.loc[(features_df['first'] >= min_x) & (features_df['first'] <= max_x) &
                                                   (features_df['mcm'] >= min_y) & (features_df['mcm'] <= max_y), 'first']
                updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(in_range_product, 'U1', 'V1')
                fig2 = plot_updated_histogram(updated_solutes_df, 'U1', 'U1 Distribution', 'U1')
                fig3 = plot_updated_histogram(updated_solvents_df, 'V1', 'V1 Distribution', 'V1')
            elif option == 'second':
                in_range_product = features_df.loc[(features_df['second'] >= min_x) & (features_df['second'] <= max_x) &
                                                   (features_df['mcm'] >= min_y) & (features_df['mcm'] <= max_y), 'second']
                updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(in_range_product, 'U2', 'V2')
                fig2 = plot_updated_histogram(updated_solutes_df, 'U2', 'U2 Distribution', 'U2')
                fig3 = plot_updated_histogram(updated_solvents_df, 'V2', 'V2 Distribution', 'V2')
            elif option == 'third':
                in_range_product = features_df.loc[(features_df['third'] >= min_x) & (features_df['third'] <= max_x) &
                                                   (features_df['mcm'] >= min_y) & (
                                                               features_df['mcm'] <= max_y), 'third']
                updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(in_range_product, 'U3', 'V3')
                fig2 = plot_updated_histogram(updated_solutes_df, 'U3', 'U3 Distribution', 'U3')
                fig3 = plot_updated_histogram(updated_solvents_df, 'V3', 'V3 Distribution', 'V3')
            elif option == 'fourth':
                in_range_product = features_df.loc[(features_df['fourth'] >= min_x) & (features_df['fourth'] <= max_x) &
                                                   (features_df['mcm'] >= min_y) & (features_df['mcm'] <= max_y), 'fourth']
                updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(in_range_product, 'U4', 'V4')
                fig2 = plot_updated_histogram(updated_solutes_df, 'U4', 'U4 Distribution', 'U4')
                fig3 = plot_updated_histogram(updated_solvents_df, 'V4', 'V4 Distribution', 'V4')

            solutes_df['in_range'] = False
            solvents_df['in_range'] = False

            return fig2, fig3

    elif (triggered_component == "negative-network-range-slider" or triggered_component == "positive-network-range-slider"):
        if option == 'first':
            filtered_data_matrix = features_df['first'][((negative_slider_value[0] <= features_df['first']) & (features_df['first'] <= negative_slider_value[1])) |
                                                        ((positive_slider_value[0] <= features_df['first']) & (features_df['first'] <= positive_slider_value[1]))].to_numpy()
            updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(filtered_data_matrix, 'U1', 'V1')
            fig2 = plot_updated_histogram(updated_solutes_df, 'U1', 'U1 Distribution', 'U1')
            fig3 = plot_updated_histogram(updated_solvents_df, 'V1', 'V1 Distribution', 'V1')
        elif option == 'second':
            filtered_data_matrix = features_df['first'][((negative_slider_value[0] <= features_df['second']) & (features_df['second'] <= negative_slider_value[1])) |
            ((positive_slider_value[0] <= features_df['second']) & (features_df['second'] <= positive_slider_value[1]))].to_numpy()
            updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(filtered_data_matrix, 'U2', 'V2')
            fig2 = plot_updated_histogram(updated_solutes_df, 'U2', 'U2 Distribution', 'U2')
            fig3 = plot_updated_histogram(updated_solvents_df, 'V2', 'V2 Distribution', 'V2')
        elif option == 'third':
            filtered_data_matrix = features_df['third'][((negative_slider_value[0] <= features_df['third']) & (features_df['third'] <= negative_slider_value[1])) |
                                                        ((positive_slider_value[0] <= features_df['third']) & (features_df['third'] <= positive_slider_value[1]))].to_numpy()
            updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(filtered_data_matrix, 'U3', 'V3')
            fig2 = plot_updated_histogram(updated_solutes_df, 'U3', 'U3 Distribution', 'U3')
            fig3 = plot_updated_histogram(updated_solvents_df, 'V3', 'V3 Distribution', 'V3')
        elif option == 'fourth':
            filtered_data_matrix = features_df['third'][((negative_slider_value[0] <= features_df['fourth']) & (features_df['fourth'] <= negative_slider_value[1])) |
                                                        ((positive_slider_value[0] <= features_df['fourth']) & (features_df['fourth'] <= positive_slider_value[1]))].to_numpy()
            updated_solutes_df, updated_solvents_df = calculate_in_range_solute_solvent_values(filtered_data_matrix, 'U4', 'V4')
            fig2 = plot_updated_histogram(updated_solutes_df, 'U4', 'U4 Distribution', 'U4')
            fig3 = plot_updated_histogram(updated_solvents_df, 'V4', 'V4 Distribution', 'V4')

        solutes_df['in_range'] = False
        solvents_df['in_range'] = False

        return fig2, fig3

@app.callback(
    [Output("network-plot", "figure", allow_duplicate=True), Output("data-store", "data", allow_duplicate=True)],
    [Input("solutes-solvents-options-config", "value"), Input("negative-network-range-slider", "value"), Input("positive-network-range-slider", "value")],
    prevent_initial_call=True)
def update_solute_solvent_network_plot(option, negative_slider_value, positive_slider_value):
    if option == 'first':
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('first', 'U1', 'V1', negative_slider_value, positive_slider_value)
    elif option == 'second':
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('second', 'U2', 'V2', negative_slider_value, positive_slider_value)
    elif option == 'third':
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('third', 'U3', 'V3', negative_slider_value, positive_slider_value)
    elif option == 'fourth':
        from_list, to_list, median_from_list, median_to_list = calculate_network_node_values('third', 'U3', 'V3', negative_slider_value, positive_slider_value)

    G = generate_network_graph(from_list, to_list, median_from_list, median_to_list)
    fig = plot_network_graph(G, negative_slider_value, positive_slider_value)

    from_list_storage = [array.tolist() for array in from_list]
    to_list_storage = [array.tolist() for array in to_list]
    median_from_list_storage = [array.tolist() for array in median_from_list]
    median_to_list_storage = [array.tolist() for array in median_to_list]
    storage_data = json.dumps([from_list_storage, to_list_storage, median_from_list_storage, median_to_list_storage])
    return fig, storage_data


@app.callback([Output("network_dist_modal", "is_open"), Output("network_modal_dist_graph","figure"), Output("network_modal_dist_table","figure")],
    [Input("network-plot", "clickData"), Input("solutes-solvents-options-config", "value"), Input("data-store", "data"), Input("network_close_modal_button", "n_clicks")],
    [State("network_dist_modal", "is_open")],
    prevent_initial_call=True)
def toggle_solute_solvent_network_modal(clickData, option, data, close_clicks, is_open):
    ctx = dash.callback_context
    triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_component == 'drop-down':
        return [dash.no_update, dash.no_update, dash.no_update]

    if triggered_component == 'data-store':
        return [dash.no_update, dash.no_update, dash.no_update]

    if triggered_component == 'network-plot' and clickData is None:
        empty_graph_figure = go.Figure()
        empty_table_figure = go.Figure()
        return False, empty_graph_figure, empty_table_figure

    if data is None:
        empty_graph_figure = go.Figure()
        empty_table_figure = go.Figure()
        return False, empty_graph_figure, empty_table_figure

    if triggered_component == 'network-plot' and clickData is not None:
        from_list, to_list, median_from_list, median_to_list = json.loads(data)
        for value in clickData.values():
            selected_node = np.float64(value[0].get('text'))
            if option == 'first':
                is_open, fig1, fig2 = plot_network_dist_data(from_list, to_list, median_from_list, median_to_list, selected_node, 'U1', 'V1')
            elif option == 'second':
                is_open, fig1, fig2 = plot_network_dist_data(from_list, to_list, median_from_list, median_to_list, selected_node, 'U2', 'V2')
            elif option == 'third':
                is_open, fig1, fig2 = plot_network_dist_data(from_list, to_list, median_from_list, median_to_list, selected_node, 'U3', 'V3')
            elif option == 'fourth':
                is_open, fig1, fig2 = plot_network_dist_data(from_list, to_list, median_from_list, median_to_list, selected_node, 'U4', 'V4')

            return is_open, fig1, fig2

    if triggered_component == 'close-modal-button':
        empty_graph_figure = go.Figure()
        empty_table_figure = go.Figure()
        return False, empty_graph_figure, empty_table_figure

    return is_open, None, None

@callback([Output("correlation-matrix", "figure"), Output("violin-plots", "figure")],
          Input("correlations-options-config", "value"))
def display_correlation_plots(option):
    selected_columns = ['first', 'second', 'third', 'fourth', 'mcm']
    correlation_matrix = features_df[selected_columns].corr()

    fig1= px.imshow(
        correlation_matrix,
        labels=dict(color="Correlation"),
        x=['U1*V1', 'U2*V2', 'U3*V3', 'U4*V4', 'MCM'],
        y=['U1*V1', 'U2*V2', 'U3*V3', 'U4*V4', 'MCM'],
        color_continuous_scale='Plasma',
        text_auto=True
    )

    fig1.update_layout(width=600, height=600)

    if option == 'solutes':
        fig2 = display_solute_violin_plots()
    elif option == 'solvents':
        fig2 = display_solvent_violin_plots()

    return fig1, fig2

@callback(
    [Output('correlations_det_modal', 'is_open'), Output('correlations_modal_det_table', 'figure')],
    [Input("correlations-options-config", "value"), Input('violin-plots', 'relayoutData'), Input('correlations_close_modal_button', 'n_clicks')],
    [State("correlations_det_modal", "is_open")],
    prevent_initial_call=True)
def display_solute_solvent_table(option, relayoutData, close_clicks, is_open):
    ctx = dash.callback_context
    triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]

    empty_table_trace = go.Table(
        header=dict(values=[], align='left', line_color='darkslategray', height=40),
        columnwidth=[4, 10, 5, 5, 5],
        cells=dict(values=[[] for _ in range(5)],
                   line_color='darkslategray', fill_color='lightcyan',
                   align='left', font_size=12, height=30)
    )
    empty_table_figure = go.Figure(data=[empty_table_trace])

    if triggered_component == 'correlations_close_modal_button':
        return False, empty_table_figure

    if relayoutData is not None:
        if relayoutData == {'autosize': True}:
            return False, empty_table_figure

    if triggered_component == 'violin-plots' and relayoutData is not None:
        if option == 'solutes':
            if any(key in relayoutData for key in ["xaxis.autorange", "xaxis2.autorange", "xaxis3.autorange", "xaxis4.autorange"]):
                fig = display_solute_violin_plots()
                return False, fig

            keys_to_check = ["xaxis.range[0]", "xaxis2.range[0]", "xaxis3.range[0]", "xaxis4.range[0]"]
            matching_key = None

            for key in keys_to_check:
                if key in relayoutData:
                    matching_key = key
                    break

            if matching_key:
                columns_to_display = ['I', 'Component Name', 'Molecular Formula', 'CAS Number', 'UNIFAC Groups']
                def plot1():
                    for key in relayoutData.keys():
                        if (key == 'xaxis.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis.range[0]'):
                            min_y = relayoutData[key]
                        if (key == 'yaxis.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solute_values = solutes_df.loc[(solutes_df['U1'] >= min_x) & (solutes_df['U1'] <= max_x) &
                                                            (solutes_df['U1'] >= min_y) & (solutes_df['U1'] <= max_y)]
                    return in_range_solute_values

                def plot2():
                    for key in relayoutData.keys():
                        if (key == 'xaxis2.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis2.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis2.range[0]'):
                            min_y = relayoutData[key]
                        if (key == 'yaxis2.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solute_values = solutes_df.loc[(solutes_df['U2'] >= min_x) & (solutes_df['U2'] <= max_x) &
                                                            (solutes_df['U2'] >= min_y) & (solutes_df['U2'] <= max_y)]
                    return in_range_solute_values

                def plot3():
                    for key in relayoutData.keys():
                        if (key == 'xaxis3.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis3.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis3.range[0]'):
                            min_y = relayoutData[key]
                        if (key == 'yaxis3.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solute_values = solutes_df.loc[(solutes_df['U3'] >= min_x) & (solutes_df['U3'] <= max_x) &
                                                            (solutes_df['U3'] >= min_y) & (solutes_df['U3'] <= max_y)]
                    return in_range_solute_values

                def plot4():
                    for key in relayoutData.keys():
                        if (key == 'xaxis4.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis4.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis4.range[0]'):
                            min_y = relayoutData[key]
                        if (key == 'yaxis4.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solute_values = solutes_df.loc[(solutes_df['U4'] >= min_x) & (solutes_df['U4'] <= max_x) &
                                                            (solutes_df['U4'] >= min_y) & (solutes_df['U4'] <= max_y)]
                    return in_range_solute_values

                actions = {
                    "xaxis.range[0]": plot1,
                    "xaxis2.range[0]": plot2,
                    "xaxis3.range[0]": plot3,
                    "xaxis4.range[0]": plot4
                }
                action_function = actions.get(matching_key, None)

                if action_function:
                   in_range_solute_values = action_function()

                fig = go.Figure(data=go.Table(
                    header=dict(values=list(columns_to_display), align='left', line_color='darkslategray', height=40),
                    columnwidth=[4, 10, 5, 5, 5],
                    cells=dict(values=[in_range_solute_values[col_name] for col_name in columns_to_display],
                               line_color='darkslategray', fill_color='lightcyan',
                               align='left', font_size=12, height=30)))

                #fig.update_layout(height=500, width=600)
                return True, fig

        if option == 'solvents':
            print('Option')
            if any(key in relayoutData for key in ["xaxis.autorange", "xaxis2.autorange", "xaxis3.autorange", "xaxis4.autorange"]):
                fig = display_solvent_violin_plots()
                return False, fig

            keys_to_check = ["xaxis.range[0]", "xaxis2.range[0]", "xaxis3.range[0]", "xaxis4.range[0]"]
            matching_key = None

            for key in keys_to_check:
                if key in relayoutData:
                    matching_key = key
                    break

            if matching_key:
                columns_to_display = ['J', 'Component Name', 'Molecular Formula', 'CAS Number', 'UNIFAC Groups']
                def plot1():
                    for key in relayoutData.keys():
                        if (key == 'xaxis.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis.range[0]'):
                            min_y = relayoutData[key]
                        if (key == 'yaxis.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solvent_values = solvents_df.loc[(solvents_df['V1'] >= min_x) & (solvents_df['V1'] <= max_x) &
                                                              (solvents_df['V1'] >= min_y) & (solvents_df['V1'] <= max_y)]
                    return in_range_solvent_values

                def plot2():
                    for key in relayoutData.keys():
                        if (key == 'xaxis2.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis2.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis2.range[0]'):
                           min_y = relayoutData[key]
                        if (key == 'yaxis2.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solvent_values = solvents_df.loc[(solvents_df['V2'] >= min_x) & (solvents_df['V2'] <= max_x) &
                                                              (solvents_df['V2'] >= min_y) & (solvents_df['V2'] <= max_y)]
                    return in_range_solvent_values

                def plot3():
                    for key in relayoutData.keys():
                        if (key == 'xaxis3.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis3.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis3.range[0]'):
                            min_y = relayoutData[key]
                        if (key == 'yaxis3.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solvent_values = solvents_df.loc[(solvents_df['V3'] >= min_x) & (solvents_df['V3'] <= max_x) &
                                                              (solvents_df['V3'] >= min_y) & (solvents_df['V3'] <= max_y)]
                    return in_range_solvent_values

                def plot4():
                    for key in relayoutData.keys():
                        if (key == 'xaxis4.range[0]'):
                            min_x = relayoutData[key]
                        if (key == 'xaxis4.range[1]'):
                            max_x = relayoutData[key]
                        if (key == 'yaxis4.range[0]'):
                            min_y = relayoutData[key]
                        if (key == 'yaxis4.range[1]'):
                            max_y = relayoutData[key]
                    in_range_solvent_values = solvents_df.loc[(solvents_df['V4'] >= min_x) & (solvents_df['V4'] <= max_x) &
                                                              (solvents_df['V4'] >= min_y) & (solvents_df['V4'] <= max_y)]
                    return in_range_solvent_values

                actions = {
                    "xaxis.range[0]": plot1,
                    "xaxis2.range[0]": plot2,
                    "xaxis3.range[0]": plot3,
                    "xaxis4.range[0]": plot4
                }
                action_function = actions.get(matching_key, None)

                if action_function:
                    in_range_solvent_values = action_function()

                fig = go.Figure(data=go.Table(
                    header=dict(values=list(columns_to_display), align='left', line_color='darkslategray', height=40),
                    columnwidth=[4, 10, 5, 5, 5],
                    cells=dict(values=[in_range_solvent_values[col_name] for col_name in columns_to_display],
                               line_color='darkslategray', fill_color='lightcyan',
                               align='left', font_size=12, height=30)))

                #fig.update_layout(height=500, width=600)
                return True, fig

@callback([Output("solute-parallel-coords-plot", "figure"), Output("solute-clustergram", "figure")],
          [Input("solute-options-config", "value")])
def display_solute_charts(option):
    solutes_df['U1_categorical'] = convert_to_categorical(solutes_df['U1'])
    solutes_df['U2_categorical'] = convert_to_categorical(solutes_df['U2'])
    solutes_df['U3_categorical'] = convert_to_categorical(solutes_df['U3'])
    solutes_df['U4_categorical'] = convert_to_categorical(solutes_df['U4'])

    solute_medians = np.median(full_mcm, axis=1)
    fig1 = px.parallel_categories(solutes_df, dimensions=['U1_categorical', 'U2_categorical', 'U3_categorical', 'U4_categorical'], color=solute_medians,
                                 labels={'U1_categorical': 'U1', 'U2_categorical': 'U2', 'U3_categorical': 'U3', 'U4_categorical': 'U4'})
    fig1.update_layout(height=600)

    if option == 'first':
        u1_values = solutes_df['U1'].to_numpy()
        first_solute_latent = np.reshape(u1_values, (24, 10))
        fig2 = dashbio.Clustergram(data=first_solute_latent, color_threshold={'row': 250, 'col': 700})

    elif option == 'second':
        u2_values = solutes_df['U2'].to_numpy()
        second_solute_latent = np.reshape(u2_values, (24, 10))
        fig2 = dashbio.Clustergram(data=second_solute_latent, color_threshold={'row': 250, 'col': 700})

    elif option == 'third':
        u3_values = solutes_df['U3'].to_numpy()
        third_solute_latent = np.reshape(u3_values, (24, 10))
        fig2 = dashbio.Clustergram(data=third_solute_latent, color_threshold={'row': 250, 'col': 700})

    elif option == 'fourth':
        u4_values = solutes_df['U4'].to_numpy()
        fourth_solute_latent = np.reshape(u4_values, (24, 10))
        fig2 = dashbio.Clustergram(data=fourth_solute_latent, color_threshold={'row': 250, 'col': 700})

    fig2.update_layout(height=600, width=1000)

    return fig1, fig2


@callback([Output("solvent-parallel-coords-plot", "figure"), Output("solvent-clustergram", "figure")],
          [Input("solvent-options-config", "value")])
def display_solvent_charts(option):
    solvents_df['V1_categorical'] = convert_to_categorical(solvents_df['V1'])
    solvents_df['V2_categorical'] = convert_to_categorical(solvents_df['V2'])
    solvents_df['V3_categorical'] = convert_to_categorical(solvents_df['V3'])
    solvents_df['V4_categorical'] = convert_to_categorical(solvents_df['V4'])

    solvent_medians = np.median(full_mcm, axis=0)
    fig1 = px.parallel_categories(solvents_df, dimensions=['V1_categorical', 'V2_categorical', 'V3_categorical', 'V4_categorical'], color=solvent_medians,
                                     labels={'V1_categorical': 'V1', 'V2_categorical': 'V2', 'V3_categorical': 'V3', 'V4_categorical': 'V4'})
    fig1.update_layout(height=600)

    if option == 'first':
        v1_values = solvents_df['V1'].to_numpy()
        first_solvent_latent = np.reshape(v1_values, (25, 10))
        fig2 = dashbio.Clustergram(data=first_solvent_latent, color_threshold={'row': 250, 'col': 700})

    elif option == 'second':
        v2_values = solvents_df['V2'].to_numpy()
        second_solvent_latent = np.reshape(v2_values, (25, 10))
        fig2 = dashbio.Clustergram(data=second_solvent_latent, color_threshold={'row': 250, 'col': 700})

    elif option == 'third':
        v3_values = solvents_df['V3'].to_numpy()
        third_solvent_latent = np.reshape(v3_values, (25, 10))
        fig2 = dashbio.Clustergram(data=third_solvent_latent, color_threshold={'row': 250, 'col': 700})

    elif option == 'fourth':
        v4_values = solvents_df['V4'].to_numpy()
        fourth_solvent_latent = np.reshape(v4_values, (25, 10))
        fig2 = dashbio.Clustergram(data=fourth_solvent_latent, color_threshold={'row': 250, 'col': 700})

    fig2.update_layout(height=600, width=1000)

    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)
