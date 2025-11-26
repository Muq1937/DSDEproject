"""
Folium map generator for Bangkok Traffy complaints.
Creates interactive maps with marker clustering for large datasets.
"""

import folium
from folium import plugins
import pandas as pd
from data_processor import TraffyDataProcessor
from pathlib import Path
import json


class FoliumMapGenerator:
    """Generate interactive Folium maps with clustering."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize map generator.

        Args:
            df: DataFrame with complaint data
        """
        self.df = df
        self.bangkok_center = [13.7563, 100.5018]
        self.color_map = self._create_color_map()

    def _create_color_map(self):
        """Create color mapping for complaint types."""
        # Get top complaint types
        top_types = self.df['primary_type'].value_counts().head(15).index.tolist()

        # Color palette
        colors = [
            'red', 'blue', 'green', 'purple', 'orange',
            'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
            'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen'
        ]

        color_map = {t: colors[i % len(colors)] for i, t in enumerate(top_types)}
        color_map['Unknown'] = 'gray'

        return color_map

    def create_clustered_map(self, sample_size: int = 10000, time_period: str = None) -> folium.Map:
        """
        Create a map with marker clustering.

        Args:
            sample_size: Maximum number of markers to display
            time_period: Optional time period filter (year_month format)

        Returns:
            Folium map object
        """
        # Filter data
        df_map = self.df.copy()
        if time_period:
            df_map = df_map[df_map['year_month'] == time_period]

        # Sample if too large
        if len(df_map) > sample_size:
            df_map = df_map.sample(n=sample_size, random_state=42)

        print(f"Creating map with {len(df_map):,} markers...")

        # Create base map
        m = folium.Map(
            location=self.bangkok_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # Add marker cluster
        marker_cluster = plugins.MarkerCluster(
            name='Complaints',
            overlay=True,
            control=True,
            icon_create_function=None
        ).add_to(m)

        # Add markers
        for idx, row in df_map.iterrows():
            # Create popup content
            popup_html = f"""
            <div style="width: 250px;">
                <h4>{row['primary_type']}</h4>
                <hr>
                <b>District:</b> {row['district']}<br>
                <b>Subdistrict:</b> {row['subdistrict']}<br>
                <b>Date:</b> {row['timestamp'].strftime('%Y-%m-%d')}<br>
                <b>Status:</b> {row['state']}<br>
                <b>Solve Days:</b> {row['solve_days']:.0f}<br>
                <b>Reopened:</b> {int(row['count_reopen'])} times<br>
                <hr>
                <small>{row['comment'][:100]}...</small>
            </div>
            """

            # Get color
            color = self.color_map.get(row['primary_type'], 'gray')

            # Create marker
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['primary_type']} - {row['district']}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(marker_cluster)

        # Add layer control
        folium.LayerControl().add_to(m)

        return m

    def create_heatmap(self, time_period: str = None) -> folium.Map:
        """
        Create a heatmap visualization.

        Args:
            time_period: Optional time period filter

        Returns:
            Folium map with heatmap layer
        """
        # Filter data
        df_heat = self.df.copy()
        if time_period:
            df_heat = df_heat[df_heat['year_month'] == time_period]

        print(f"Creating heatmap with {len(df_heat):,} points...")

        # Create base map
        m = folium.Map(
            location=self.bangkok_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # Prepare heat data
        heat_data = [[row['lat'], row['lon']] for idx, row in df_heat.iterrows()]

        # Add heatmap
        plugins.HeatMap(
            heat_data,
            name='Complaint Density',
            min_opacity=0.3,
            max_zoom=13,
            radius=15,
            blur=20,
            gradient={
                0.0: 'blue',
                0.5: 'lime',
                0.7: 'yellow',
                1.0: 'red'
            }
        ).add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        return m

    def create_time_series_map(self, max_periods: int = 12) -> folium.Map:
        """
        Create a map with time-based animation using TimestampedGeoJson.

        Args:
            max_periods: Maximum number of time periods to include

        Returns:
            Folium map with time animation
        """
        # Get time periods
        periods = sorted(self.df['year_month'].unique())[-max_periods:]

        # Create base map
        m = folium.Map(
            location=self.bangkok_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # Prepare features for each time period
        features = []

        for period in periods:
            df_period = self.df[self.df['year_month'] == period].sample(n=min(500, len(self.df)), random_state=42)

            for idx, row in df_period.iterrows():
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [row['lon'], row['lat']]
                    },
                    'properties': {
                        'time': f"{period}-01",  # Use first day of month
                        'style': {'color': self.color_map.get(row['primary_type'], 'gray')},
                        'icon': 'circle',
                        'iconstyle': {
                            'fillColor': self.color_map.get(row['primary_type'], 'gray'),
                            'fillOpacity': 0.6,
                            'stroke': 'true',
                            'radius': 5
                        },
                        'popup': f"<b>{row['primary_type']}</b><br>{row['district']}<br>{period}"
                    }
                }
                features.append(feature)

        # Create TimestampedGeoJson
        plugins.TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': features
            },
            period='P1M',  # 1 month period
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=2,
            loop_button=True,
            time_slider_drag_update=True
        ).add_to(m)

        return m

    def create_choropleth_map(self, district_stats: pd.DataFrame = None) -> folium.Map:
        """
        Create a choropleth map by district (requires GeoJSON).

        Args:
            district_stats: DataFrame with district-level statistics

        Returns:
            Folium map with choropleth layer
        """
        # Aggregate by district
        if district_stats is None:
            district_stats = self.df.groupby('district').agg({
                'lat': 'count',  # complaint count
                'solve_days': 'mean'
            }).reset_index()
            district_stats.columns = ['district', 'complaint_count', 'avg_solve_days']

        # Create base map
        m = folium.Map(
            location=self.bangkok_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )

        # Note: To create actual choropleth, you need Bangkok district GeoJSON
        # This is a placeholder showing district markers instead

        # Calculate district centers
        district_centers = self.df.groupby('district').agg({
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()

        district_data = district_centers.merge(district_stats, on='district')

        # Normalize for color
        max_count = district_data['complaint_count'].max()

        for idx, row in district_data.iterrows():
            # Calculate color intensity
            intensity = row['complaint_count'] / max_count

            # Create circle marker
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10 + (intensity * 30),  # Size based on count
                popup=f"""
                <b>{row['district']}</b><br>
                Complaints: {row['complaint_count']:,}<br>
                Avg Solve Days: {row['avg_solve_days']:.1f}
                """,
                tooltip=f"{row['district']}: {row['complaint_count']:,} complaints",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.3 + (intensity * 0.5)
            ).add_to(m)

        return m

    def save_map(self, map_obj: folium.Map, filename: str, output_dir: str = "maps"):
        """
        Save map to HTML file.

        Args:
            map_obj: Folium map object
            filename: Output filename
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        filepath = output_path / filename
        map_obj.save(str(filepath))
        print(f"Map saved to: {filepath}")

        return filepath


def generate_all_maps(csv_path: str, sample_frac: float = 0.1):
    """
    Generate all map types.

    Args:
        csv_path: Path to CSV file
        sample_frac: Fraction of data to use
    """
    # Load data
    processor = TraffyDataProcessor(csv_path)
    df = processor.load_data_chunked(sample_frac=sample_frac)

    # Create generator
    generator = FoliumMapGenerator(df)

    # Generate maps
    print("\n=== Generating Maps ===\n")

    # 1. Clustered map
    print("1. Creating clustered marker map...")
    clustered_map = generator.create_clustered_map(sample_size=5000)
    generator.save_map(clustered_map, "clustered_map.html")

    # 2. Heatmap
    print("\n2. Creating heatmap...")
    heatmap = generator.create_heatmap()
    generator.save_map(heatmap, "heatmap.html")

    # 3. Time series map
    print("\n3. Creating time series animation...")
    time_map = generator.create_time_series_map()
    generator.save_map(time_map, "time_series_map.html")

    # 4. District map
    print("\n4. Creating district-level map...")
    district_map = generator.create_choropleth_map()
    generator.save_map(district_map, "district_map.html")

    print("\n=== All maps generated successfully! ===")
    print("Maps saved in 'maps/' directory")


if __name__ == "__main__":
    # Generate all maps
    generate_all_maps("bangkok_traffy_30.csv", sample_frac=0.1)
