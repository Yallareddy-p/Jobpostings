import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from data_analysis import (
    load_data, clean_job_titles, extract_skills, 
    generate_analytics, generate_time_series, generate_skill_network
)

# Load and process data
df = load_data()
df = clean_job_titles(df)
df = extract_skills(df)
title_counts, skill_counts, location_counts, company_counts = generate_analytics(df)
time_series_df = generate_time_series()
edge_trace, node_trace = generate_skill_network(df)

# Convert to DataFrame for Plotly
title_df = pd.DataFrame({'Job Title': title_counts.index, 'Count': title_counts.values})
skill_df = pd.DataFrame({'Skill': skill_counts.index, 'Count': skill_counts.values})
location_df = pd.DataFrame({'Location': location_counts.index, 'Count': location_counts.values})
company_df = pd.DataFrame({'Company': company_counts.index, 'Count': company_counts.values})

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout with filters
app.layout = html.Div([
    html.H1('ML Job Analytics Dashboard', style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    # Filters Section
    html.Div([
        html.Div([
            html.H3('Filters', style={'color': '#2c3e50'}),
            html.Label('Select Job Title:'),
            dcc.Dropdown(
                id='title-filter',
                options=[{'label': title, 'value': title} for title in title_df['Job Title']],
                value=title_df['Job Title'].tolist(),
                multi=True
            ),
            html.Br(),
            html.Label('Select Location:'),
            dcc.Dropdown(
                id='location-filter',
                options=[{'label': loc, 'value': loc} for loc in location_df['Location']],
                value=location_df['Location'].tolist(),
                multi=True
            ),
            html.Br(),
            html.Label('Select Company:'),
            dcc.Dropdown(
                id='company-filter',
                options=[{'label': comp, 'value': comp} for comp in company_df['Company']],
                value=company_df['Company'].tolist(),
                multi=True
            ),
            html.Br(),
            html.Label('Date Range:'),
            dcc.DatePickerRange(
                id='date-range',
                start_date=time_series_df['Month'].min(),
                end_date=time_series_df['Month'].max()
            )
        ], style={'width': '20%', 'float': 'left', 'padding': '20px'}),
        
        # Main Content
        html.Div([
            # Job Title Distribution
            html.Div([
                html.H2('Job Title Distribution', style={'color': '#2c3e50'}),
                dcc.Graph(
                    id='title-graph',
                    figure=px.bar(title_df, 
                                x='Job Title', 
                                y='Count', 
                                title='Job Title Counts',
                                template='plotly_white')
                )
            ]),
            
            # Skill Network
            html.Div([
                html.H2('Skill Co-occurrence Network', style={'color': '#2c3e50'}),
                dcc.Graph(
                    id='skill-network',
                    figure=go.Figure(
                        data=edge_trace + [go.Scatter(**node_trace)],
                        layout=go.Layout(
                            title='Skill Co-occurrence Network',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                    )
                )
            ]),
            
            # Skill Distribution
            html.Div([
                html.H2('Skill Distribution', style={'color': '#2c3e50'}),
                dcc.Graph(
                    id='skill-graph',
                    figure=px.bar(skill_df, 
                                x='Skill', 
                                y='Count', 
                                title='Skill Counts',
                                template='plotly_white')
                )
            ]),
            
            # Geographic Distribution
            html.Div([
                html.H2('Geographic Distribution', style={'color': '#2c3e50'}),
                dcc.Graph(
                    id='location-graph',
                    figure=px.bar(location_df, 
                                x='Location', 
                                y='Count', 
                                title='Location Counts',
                                template='plotly_white')
                )
            ]),
            
            # Company Hiring Behavior
            html.Div([
                html.H2('Company Hiring Behavior', style={'color': '#2c3e50'}),
                dcc.Graph(
                    id='company-graph',
                    figure=px.bar(company_df, 
                                x='Company', 
                                y='Count', 
                                title='Company Counts',
                                template='plotly_white')
                )
            ]),
            
            # Job Posting Trends
            html.Div([
                html.H2('Job Posting Trends (Last 12 Months)', style={'color': '#2c3e50'}),
                dcc.Graph(
                    id='time-series-graph',
                    figure=px.line(time_series_df, 
                                 x='Month', 
                                 y='Job Count', 
                                 title='Monthly Job Posting Counts',
                                 template='plotly_white')
                )
            ])
        ], style={'width': '75%', 'float': 'right', 'padding': '20px'})
    ]),
    
    # Download Button
    html.Div([
        html.Button('Download Data', id='download-button', 
                   style={'backgroundColor': '#2c3e50', 'color': 'white', 'padding': '10px'}),
        dcc.Download(id='download-dataframe-csv')
    ], style={'clear': 'both', 'padding': '20px'})
])

# Callback for filtering data
@app.callback(
    [Output('title-graph', 'figure'),
     Output('location-graph', 'figure'),
     Output('company-graph', 'figure'),
     Output('time-series-graph', 'figure')],
    [Input('title-filter', 'value'),
     Input('location-filter', 'value'),
     Input('company-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_graphs(selected_titles, selected_locations, selected_companies, start_date, end_date):
    # Filter data based on selections
    filtered_title_df = title_df[title_df['Job Title'].isin(selected_titles)]
    filtered_location_df = location_df[location_df['Location'].isin(selected_locations)]
    filtered_company_df = company_df[company_df['Company'].isin(selected_companies)]
    filtered_time_df = time_series_df[
        (time_series_df['Month'] >= start_date) & 
        (time_series_df['Month'] <= end_date)
    ]
    
    # Create updated figures
    title_fig = px.bar(filtered_title_df, x='Job Title', y='Count', 
                      title='Job Title Counts', template='plotly_white')
    location_fig = px.bar(filtered_location_df, x='Location', y='Count', 
                         title='Location Counts', template='plotly_white')
    company_fig = px.bar(filtered_company_df, x='Company', y='Count', 
                        title='Company Counts', template='plotly_white')
    time_fig = px.line(filtered_time_df, x='Month', y='Job Count', 
                      title='Monthly Job Posting Counts', template='plotly_white')
    
    return title_fig, location_fig, company_fig, time_fig

# Callback for downloading data
@app.callback(
    Output('download-dataframe-csv', 'data'),
    Input('download-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_data(n_clicks):
    return dcc.send_data_frame(df.to_csv, 'ml_job_data.csv')

# For local development
if __name__ == '__main__':
    app.run_server(debug=True)
else:
    # For PythonAnywhere deployment
    application = app.server 