"""
Interactive Plotly Dash dashboard for Bangkok Traffy complaints.
Features time-slider, animated maps, and interactive visualizations.
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from data_processor import TraffyDataProcessor
from datetime import datetime
import json


class TraffyDashboard:
    """Interactive dashboard for Bangkok Traffy complaint visualization."""

    def __init__(self, csv_path: str, sample_frac: float = 0.1):
        """
        Initialize dashboard.

        Args:
            csv_path: Path to the CSV file
            sample_frac: Fraction of data to use (for large files)
        """
        self.processor = TraffyDataProcessor(csv_path)
        print(f"Loading data (sampling {sample_frac*100}% for performance)...")
        self.df = self.processor.load_data_chunked(sample_frac=sample_frac)

        # Get time range
        self.date_range = sorted(self.df['year_month'].unique())

        # Bangkok center
        self.bangkok_center = {'lat': 13.7563, 'lon': 100.5018}

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )

        # Add custom CSS to prevent chart expansion
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .dash-graph {
                        max-width: 100%;
                        overflow: hidden;
                    }
                    .plotly-graph-div {
                        max-width: 100% !important;
                        height: 100% !important;
                    }
                    .card {
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        border-radius: 8px;
                    }
                    .card-body {
                        padding: 1rem;
                    }
                    .tab-content {
                        padding: 1rem 0;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🗺️ Bangkok Traffy Complaints Dashboard",
                           className="text-center mb-4 mt-4"),
                    html.Hr()
                ])
            ]),

            # Summary stats row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(self.df):,}", className="card-title"),
                            html.P("Total Complaints", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{self.df['district'].nunique()}", className="card-title"),
                            html.P("Districts", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{self.df['solve_days'].mean():.1f}", className="card-title"),
                            html.P("Avg Solve Days", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(self.df['primary_type'].mode()[0], className="card-title"),
                            html.P("Top Complaint Type", className="card-text")
                        ])
                    ])
                ], width=3),
            ], className="mb-4"),

            # Main content tabs
            dbc.Tabs([
                # Tab 1: Geospatial with Time Slider
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Geospatial Distribution Over Time", className="mt-3"),
                            html.P("Use the slider to explore complaints evolution over time"),

                            # Time slider
                            html.Div([
                                html.Label("Select Time Period:"),
                                dcc.Slider(
                                    id='time-slider',
                                    min=0,
                                    max=len(self.date_range) - 1,
                                    value=len(self.date_range) - 1,
                                    marks={i: date for i, date in enumerate(self.date_range[::max(1, len(self.date_range)//10)])},
                                    step=1,
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ], className="mb-4"),

                            # Animation controls
                            html.Div([
                                dbc.Button("▶ Play Animation", id="play-button", color="primary", className="me-2"),
                                dbc.Button("⏸ Pause", id="pause-button", color="secondary", className="me-2"),
                                dcc.Interval(id='interval-component', interval=1000, n_intervals=0, disabled=True)
                            ], className="mb-3"),

                            # Map
                            dcc.Graph(id='geo-map', style={'height': '600px'}),

                        ], width=12)
                    ])
                ], label="🗺️ Geospatial View"),

                # Tab 2: Time Series Analysis
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Time Series Analysis", className="mt-3"),

                            # Complaint type selector
                            html.Div([
                                html.Label("Filter by Complaint Type:"),
                                dcc.Dropdown(
                                    id='type-dropdown',
                                    options=[{'label': 'All Types', 'value': 'ALL'}] +
                                           [{'label': t, 'value': t} for t in self.df['primary_type'].value_counts().head(20).index],
                                    value='ALL',
                                    clearable=False
                                )
                            ], className="mb-3"),

                            dcc.Graph(id='time-series-graph', style={'height': '450px'}),

                            html.H4("District-wise Distribution", className="mt-4"),
                            dcc.Graph(id='district-bar-chart', style={'height': '450px'}),

                        ], width=12)
                    ])
                ], label="📊 Time Series"),

                # Tab 3: Heatmap
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Complaint Density Heatmap", className="mt-3"),
                            html.P("Density heatmap showing complaint hotspots"),
                            dcc.Graph(id='heatmap', style={'height': '700px'}),
                        ], width=12)
                    ])
                ], label="🔥 Heatmap"),

                # Tab 4: Statistics
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Top Complaint Types", className="mt-3"),
                            dcc.Graph(id='type-pie-chart', style={'height': '450px'}),
                        ], width=6),
                        dbc.Col([
                            html.H4("Complaint Resolution Status", className="mt-3"),
                            dcc.Graph(id='state-pie-chart', style={'height': '450px'}),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H4("Solution Time Distribution", className="mt-4"),
                            dcc.Graph(id='solve-time-histogram', style={'height': '400px'}),
                        ], width=12)
                    ])
                ], label="📈 Statistics"),
            ]),

            # Footer
            html.Hr(),
            html.Footer([
                html.P("Bangkok Traffy Complaints Dashboard | Data Source: Traffy Fondue",
                      className="text-center text-muted")
            ])

        ], fluid=True)

    def _setup_callbacks(self):
        """Setup interactive callbacks."""

        # Time slider animation
        @self.app.callback(
            [Output('time-slider', 'value'),
             Output('interval-component', 'disabled')],
            [Input('interval-component', 'n_intervals'),
             Input('play-button', 'n_clicks'),
             Input('pause-button', 'n_clicks')],
            [State('time-slider', 'value')]
        )
        def animate_slider(n_intervals, play_clicks, pause_clicks, current_value):
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_value, True

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'play-button':
                return current_value, False
            elif trigger_id == 'pause-button':
                return current_value, True
            elif trigger_id == 'interval-component':
                next_value = (current_value + 1) % len(self.date_range)
                return next_value, False

            return current_value, True

        # Geospatial map update
        @self.app.callback(
            Output('geo-map', 'figure'),
            [Input('time-slider', 'value')]
        )
        def update_map(time_idx):
            selected_period = self.date_range[time_idx]
            df_filtered = self.df[self.df['year_month'] == selected_period]

            # Sample if too many points
            if len(df_filtered) > 5000:
                df_filtered = df_filtered.sample(n=5000, random_state=42)

            fig = px.scatter_mapbox(
                df_filtered,
                lat='lat',
                lon='lon',
                color='primary_type',
                hover_data={
                    'primary_type': True,
                    'district': True,
                    'solve_days': ':.1f',
                    'lat': ':.4f',
                    'lon': ':.4f'
                },
                zoom=10,
                center=self.bangkok_center,
                mapbox_style='carto-positron',
                title=f'Complaints in {selected_period} (showing {len(df_filtered):,} complaints)'
            )

            fig.update_layout(
                height=600,
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                showlegend=True,
                autosize=True
            )

            return fig

        # Time series graph
        @self.app.callback(
            Output('time-series-graph', 'figure'),
            [Input('type-dropdown', 'value')]
        )
        def update_time_series(selected_type):
            if selected_type == 'ALL':
                df_ts = self.df.groupby('year_month').size().reset_index(name='count')
                title = 'All Complaint Types Over Time'
            else:
                df_filtered = self.df[self.df['primary_type'] == selected_type]
                df_ts = df_filtered.groupby('year_month').size().reset_index(name='count')
                title = f'{selected_type} Complaints Over Time'

            fig = px.line(
                df_ts,
                x='year_month',
                y='count',
                title=title,
                labels={'year_month': 'Time Period', 'count': 'Number of Complaints'}
            )

            fig.update_layout(
                height=450,
                xaxis_tickangle=-45,
                margin=dict(l=50, r=20, t=50, b=80),
                autosize=True
            )
            return fig

        # District bar chart
        @self.app.callback(
            Output('district-bar-chart', 'figure'),
            [Input('type-dropdown', 'value')]
        )
        def update_district_chart(selected_type):
            if selected_type == 'ALL':
                df_dist = self.df['district'].value_counts().head(15).reset_index()
            else:
                df_filtered = self.df[self.df['primary_type'] == selected_type]
                df_dist = df_filtered['district'].value_counts().head(15).reset_index()

            df_dist.columns = ['district', 'count']

            fig = px.bar(
                df_dist,
                x='district',
                y='count',
                title='Top 15 Districts by Complaint Count',
                labels={'district': 'District', 'count': 'Number of Complaints'}
            )

            fig.update_layout(
                height=450,
                xaxis_tickangle=-45,
                margin=dict(l=50, r=20, t=50, b=80),
                autosize=True
            )
            return fig

        # Heatmap
        @self.app.callback(
            Output('heatmap', 'figure'),
            [Input('time-slider', 'value')]
        )
        def update_heatmap(time_idx):
            selected_period = self.date_range[time_idx]
            df_filtered = self.df[self.df['year_month'] == selected_period]

            fig = px.density_mapbox(
                df_filtered,
                lat='lat',
                lon='lon',
                radius=10,
                zoom=10,
                center=self.bangkok_center,
                mapbox_style='carto-positron',
                title=f'Complaint Density Heatmap - {selected_period}'
            )

            fig.update_layout(
                height=700,
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                autosize=True
            )

            return fig

        # Type pie chart
        @self.app.callback(
            Output('type-pie-chart', 'figure'),
            [Input('time-slider', 'value')]
        )
        def update_type_pie(time_idx):
            type_counts = self.df['primary_type'].value_counts().head(10)

            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title='Top 10 Complaint Types'
            )

            fig.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                ),
                autosize=True
            )
            return fig

        # State pie chart
        @self.app.callback(
            Output('state-pie-chart', 'figure'),
            [Input('time-slider', 'value')]
        )
        def update_state_pie(time_idx):
            state_counts = self.df['state'].value_counts()

            fig = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                title='Complaint Resolution Status'
            )

            fig.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                ),
                autosize=True
            )
            return fig

        # Solve time histogram
        @self.app.callback(
            Output('solve-time-histogram', 'figure'),
            [Input('type-dropdown', 'value')]
        )
        def update_histogram(selected_type):
            if selected_type == 'ALL':
                df_hist = self.df[self.df['solve_days'] < 200]  # Filter outliers
            else:
                df_hist = self.df[(self.df['primary_type'] == selected_type) &
                                 (self.df['solve_days'] < 200)]

            fig = px.histogram(
                df_hist,
                x='solve_days',
                nbins=50,
                title='Distribution of Resolution Time (Days)',
                labels={'solve_days': 'Days to Solve', 'count': 'Frequency'}
            )

            fig.update_layout(
                height=400,
                margin=dict(l=50, r=20, t=50, b=50),
                autosize=True
            )
            return fig

    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        print(f"\n{'='*60}")
        print(f"Starting Bangkok Traffy Dashboard...")
        print(f"Open your browser and navigate to: http://localhost:{port}")
        print(f"{'='*60}\n")
        self.app.run(debug=debug, port=port, host='0.0.0.0')


if __name__ == "__main__":
    # Create and run dashboard
    # Use sample_frac=1.0 for full dataset (may be slow with 900MB file)
    # Use sample_frac=0.1 for 10% sample (recommended for initial testing)
    dashboard = TraffyDashboard(
        csv_path="bangkok_traffy_30.csv",
        sample_frac=0.1  # Adjust based on your needs
    )
    dashboard.run(debug=True, port=8050)
