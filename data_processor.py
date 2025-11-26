"""
Data processing module for Bangkok Traffy complaint data.
Handles large CSV files (900MB+) efficiently using chunked processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class TraffyDataProcessor:
    """Process large Bangkok Traffy complaint datasets efficiently."""

    def __init__(self, csv_path: str, cache_dir: str = "cache"):
        """
        Initialize data processor.

        Args:
            csv_path: Path to the CSV file
            cache_dir: Directory for caching processed data
        """
        self.csv_path = Path(csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Bangkok bounds (approximate)
        self.BANGKOK_BOUNDS = {
            'lat_min': 13.5,
            'lat_max': 14.0,
            'lon_min': 100.3,
            'lon_max': 100.9
        }

    def parse_type_field(self, type_str: str) -> List[str]:
        """Parse the type field which can be a set-like string."""
        if pd.isna(type_str) or type_str == '{}':
            return []
        try:
            # Remove curly braces and split by comma
            type_str = str(type_str).strip('{}')
            if not type_str:
                return []
            # Split and clean
            types = [t.strip() for t in type_str.split(',')]
            return types
        except:
            return []

    def load_data_chunked(self, chunksize: int = 50000, sample_frac: float = None) -> pd.DataFrame:
        """
        Load data in chunks for memory efficiency.

        Args:
            chunksize: Number of rows to process at a time
            sample_frac: If provided, randomly sample this fraction of data (e.g., 0.1 for 10%)

        Returns:
            DataFrame with processed data
        """
        print(f"Loading data from {self.csv_path}...")

        # Check for cached processed data
        cache_file = self.cache_dir / f"{self.csv_path.stem}_processed.parquet"
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
            if sample_frac:
                df = df.sample(frac=sample_frac, random_state=42)
            return df

        chunks = []
        total_rows = 0

        # Define columns to load
        usecols = [
            'type', 'organization', 'comment', 'coords', 'address',
            'subdistrict', 'district', 'province', 'timestamp',
            'count_reopen', 'last_activity', 'lon', 'lat', 'solve_days',
            'state_กำลังดำเนินการ', 'state_รอรับเรื่อง', 'state_เสร็จสิ้น'
        ]

        try:
            for chunk in pd.read_csv(
                self.csv_path,
                chunksize=chunksize,
                low_memory=False,
                usecols=lambda x: x in usecols or x == ''  # Handle unnamed first column
            ):
                # Drop unnamed columns
                chunk = chunk.loc[:, ~chunk.columns.str.contains('^Unnamed')]

                # Sample if requested
                if sample_frac:
                    chunk = chunk.sample(frac=sample_frac, random_state=42)

                # Filter to Bangkok bounds
                chunk = chunk[
                    (chunk['lat'] >= self.BANGKOK_BOUNDS['lat_min']) &
                    (chunk['lat'] <= self.BANGKOK_BOUNDS['lat_max']) &
                    (chunk['lon'] >= self.BANGKOK_BOUNDS['lon_min']) &
                    (chunk['lon'] <= self.BANGKOK_BOUNDS['lon_max'])
                ]

                chunks.append(chunk)
                total_rows += len(chunk)
                print(f"  Processed {total_rows:,} rows...", end='\r')

            print(f"\nCombining {len(chunks)} chunks...")
            df = pd.concat(chunks, ignore_index=True)

            # Process data
            df = self._process_dataframe(df)

            # Cache the processed data
            print(f"Caching processed data to {cache_file}")
            df.to_parquet(cache_file, index=False, compression='snappy')

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe with type conversions and cleaning."""
        print("Processing dataframe...")

        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['last_activity'] = pd.to_datetime(df['last_activity'], errors='coerce')

        # Extract date components for time-based analysis
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
        df['date'] = df['timestamp'].dt.date

        # Parse complaint types
        df['complaint_types'] = df['type'].apply(self.parse_type_field)
        df['num_types'] = df['complaint_types'].apply(len)

        # Primary type (first type listed)
        df['primary_type'] = df['complaint_types'].apply(
            lambda x: x[0] if len(x) > 0 else 'Unknown'
        )

        # Determine complaint state
        state_cols = ['state_กำลังดำเนินการ', 'state_รอรับเรื่อง', 'state_เสร็จสิ้น']
        df['state'] = 'Unknown'
        for col in state_cols:
            if col in df.columns:
                df.loc[df[col] == 1.0, 'state'] = col.replace('state_', '')

        # Handle missing values
        df['solve_days'] = pd.to_numeric(df['solve_days'], errors='coerce')
        df['count_reopen'] = pd.to_numeric(df['count_reopen'], errors='coerce').fillna(0)

        # Drop rows with missing critical data
        df = df.dropna(subset=['lat', 'lon', 'timestamp'])

        print(f"Final dataset: {len(df):,} rows")
        return df

    def aggregate_by_time(self, df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
        """
        Aggregate complaints by time period.

        Args:
            df: Input dataframe
            freq: Pandas frequency string ('D' for day, 'W' for week, 'M' for month)

        Returns:
            Aggregated dataframe
        """
        df_time = df.set_index('timestamp').resample(freq).agg({
            'lat': 'count',  # Count as proxy
            'primary_type': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
            'solve_days': 'mean',
            'count_reopen': 'sum'
        }).reset_index()

        df_time.rename(columns={'lat': 'complaint_count'}, inplace=True)
        return df_time

    def aggregate_spatial(self, df: pd.DataFrame, grid_size: float = 0.01) -> pd.DataFrame:
        """
        Aggregate complaints spatially using a grid.

        Args:
            df: Input dataframe
            grid_size: Size of grid cells in degrees (0.01 ≈ 1.1 km)

        Returns:
            Aggregated dataframe with grid cells
        """
        df = df.copy()

        # Create grid coordinates
        df['lat_grid'] = (df['lat'] / grid_size).round() * grid_size
        df['lon_grid'] = (df['lon'] / grid_size).round() * grid_size

        # Aggregate by grid cell
        df_spatial = df.groupby(['lat_grid', 'lon_grid']).agg({
            'lat': 'count',
            'primary_type': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
            'solve_days': 'mean'
        }).reset_index()

        df_spatial.rename(columns={'lat': 'complaint_count'}, inplace=True)
        return df_spatial

    def get_complaint_type_network(self, df: pd.DataFrame) -> Dict:
        """
        Build network of co-occurring complaint types.

        Returns:
            Dictionary with nodes and edges for network visualization
        """
        from itertools import combinations
        from collections import defaultdict, Counter

        # Count co-occurrences
        edge_counts = Counter()
        node_counts = Counter()

        for types in df['complaint_types']:
            if len(types) > 0:
                # Count individual types
                for t in types:
                    node_counts[t] += 1

                # Count pairs (co-occurrences)
                if len(types) > 1:
                    for pair in combinations(sorted(types), 2):
                        edge_counts[pair] += 1

        # Build network structure
        nodes = [
            {'id': node, 'count': count, 'label': node}
            for node, count in node_counts.most_common(50)  # Top 50 types
        ]

        # Filter edges to only include nodes in top 50
        top_types = set([n['id'] for n in nodes])
        edges = [
            {'source': pair[0], 'target': pair[1], 'weight': count}
            for pair, count in edge_counts.items()
            if pair[0] in top_types and pair[1] in top_types and count > 5
        ]

        return {'nodes': nodes, 'edges': edges}

    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for the dataset."""
        stats = {
            'total_complaints': len(df),
            'date_range': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d')
            },
            'top_types': df['primary_type'].value_counts().head(10).to_dict(),
            'avg_solve_days': df['solve_days'].mean(),
            'reopen_rate': (df['count_reopen'] > 0).mean(),
            'districts': df['district'].nunique(),
            'state_distribution': df['state'].value_counts().to_dict()
        }
        return stats


if __name__ == "__main__":
    # Example usage
    processor = TraffyDataProcessor("bangkok_traffy_30.csv")

    # For testing with small sample
    df = processor.load_data_chunked(sample_frac=0.1)

    print("\n=== Summary Statistics ===")
    stats = processor.get_summary_stats(df)
    print(f"Total complaints: {stats['total_complaints']:,}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"\nTop 5 complaint types:")
    for ctype, count in list(stats['top_types'].items())[:5]:
        print(f"  {ctype}: {count:,}")
