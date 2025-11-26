# Bangkok Traffy Complaints Visualization Dashboard

An interactive geospatial dashboard for visualizing and analyzing Bangkok Traffy complaint data. Features include time-series animation, heatmaps, network graphs, and comprehensive analytics.

## Features

### 🗺️ Interactive Geospatial Dashboard
- **Time-slider visualization** with complaint evolution over time
- **Animated playback** to watch trends unfold
- **Heatmap layers** showing complaint density
- **Interactive filtering** by complaint type and district
- **Real-time statistics** and metrics

### 🔗 Network Graph Analysis
- **Complaint type relationships** showing co-occurrence patterns
- **Community detection** to identify related complaint clusters
- **Centrality metrics** to find most important complaint types
- **Interactive network visualization** with Plotly

### 🗺️ Folium Maps
- **Marker clustering** for efficient large dataset visualization
- **Multiple map types**: clustered markers, heatmaps, time-series animations
- **District-level aggregation** maps
- **Export to standalone HTML** files

### 📊 Data Processing
- **Efficient chunked loading** for large CSV files (handles 900MB+)
- **Smart caching** with Parquet for faster reloads
- **Bangkok-specific filtering** with geographic bounds
- **Aggregation utilities** for time-series and spatial analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for full 900MB dataset)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Setup

1. **Clone the repository**
   ```bash
   cd DSDEproject
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv

   # On Linux/Mac:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place your Bangkok Traffy CSV file in the project directory
   - Example: `bangkok_traffy_30.csv` (small sample included)
   - For 900MB file: ensure sufficient disk space for cache

### Quick Start with Helper Scripts

For easier setup, use the provided helper scripts:

**On Linux/Mac:**
```bash
./run.sh
```

**On Windows:**
```cmd
run.bat
```

These scripts will:
- Automatically create virtual environment if needed
- Install dependencies on first run
- Prompt you to select visualization mode
- Ask for sample size and CSV filename
- Handle activation/deactivation automatically

## Usage

### Quick Start

Run the interactive dashboard with 10% sample (recommended for testing):
```bash
python app.py dashboard --csv bangkok_traffy_30.csv --sample 0.1
```

Then open your browser to: `http://localhost:8050`

### Command Line Interface

The application supports multiple modes:

#### 1. Interactive Dashboard
```bash
# Run with 10% sample
python app.py dashboard --csv your_file.csv --sample 0.1

# Run with full dataset
python app.py dashboard --csv your_file.csv --sample 1.0

# Run on custom port
python app.py dashboard --csv your_file.csv --sample 0.1 --port 8080
```

#### 2. Generate Folium Maps
```bash
# Generate all static map types
python app.py maps --csv your_file.csv --sample 0.1
```

This creates maps in the `maps/` directory:
- `clustered_map.html` - Marker clustering visualization
- `heatmap.html` - Density heatmap
- `time_series_map.html` - Time-based animation
- `district_map.html` - District-level aggregation

#### 3. Generate Network Graph
```bash
# Create complaint type network
python app.py network --csv your_file.csv --sample 0.1
```

Outputs:
- `network_graph.html` - Interactive network visualization
- `network_stats.txt` - Detailed network statistics

#### 4. Generate Everything
```bash
# Create all visualizations at once
python app.py all --csv your_file.csv --sample 0.1
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--csv` | Path to CSV file | `bangkok_traffy_30.csv` |
| `--sample` | Fraction of data to use (0.0-1.0) | `0.1` |
| `--port` | Dashboard server port | `8050` |
| `--no-cache` | Clear cache and reload from CSV | False |

### Performance Tuning

For **900MB CSV files**, start with small samples and increase gradually:

```bash
# Test with 1% (fast, good for development)
python app.py dashboard --csv large_file.csv --sample 0.01

# Standard usage with 10% (balanced)
python app.py dashboard --csv large_file.csv --sample 0.1

# Production with 50% (slower but comprehensive)
python app.py dashboard --csv large_file.csv --sample 0.5

# Full dataset (requires significant RAM)
python app.py dashboard --csv large_file.csv --sample 1.0
```

**Memory Guidelines:**
- 1% sample: ~500MB RAM
- 10% sample: ~2GB RAM
- 50% sample: ~6GB RAM
- 100% (900MB file): ~10GB RAM

## Dashboard Features

### Tab 1: Geospatial View
- Time slider to explore complaints over time
- Play/Pause animation controls
- Interactive map with complaint markers
- Color-coded by complaint type
- Hover for detailed information

### Tab 2: Time Series
- Line charts showing complaint trends
- Filter by specific complaint types
- District-level bar charts
- Interactive legends

### Tab 3: Heatmap
- Density visualization of complaint hotspots
- Time-period filtering
- Zoom and pan controls

### Tab 4: Statistics
- Pie charts for complaint types and status
- Solution time distribution histogram
- Summary metrics cards

## Data Format

The application expects CSV files with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `type` | Complaint type(s) in set format | Yes |
| `lat` | Latitude coordinate | Yes |
| `lon` | Longitude coordinate | Yes |
| `timestamp` | Complaint timestamp | Yes |
| `district` | Bangkok district name | No |
| `subdistrict` | Subdistrict name | No |
| `solve_days` | Days to resolution | No |
| `state_*` | Status columns | No |
| `comment` | Complaint description | No |

Example row:
```csv
type,lat,lon,timestamp,district,solve_days
"{น้ำท่วม,ร้องเรียน}",13.67891,100.66709,2021-09-19 14:56:08,ประเวศ,176
```

## Architecture

```
DSDEproject/
├── app.py                  # Main application entry point
├── data_processor.py       # Data loading and processing
├── dashboard.py            # Plotly Dash interactive dashboard
├── folium_map.py          # Folium map generation
├── network_graph.py       # Network graph analysis
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── cache/                # Cached processed data (auto-generated)
└── maps/                 # Generated Folium maps (auto-generated)
```

## Caching System

The application uses intelligent caching to speed up repeated loads:

- **Cached files**: Stored in `cache/` as Parquet format
- **Auto-detection**: Automatically uses cache if available
- **Clear cache**: Use `--no-cache` flag to force reload
- **Benefits**: 10-50x faster loading for subsequent runs

## Network Graph Insights

The network graph reveals:

1. **Co-occurring complaints**: Which complaint types often appear together
2. **Communities**: Groups of related complaint types
3. **Central nodes**: Most influential complaint types
4. **Connection strength**: Frequency of co-occurrence

### Metrics Calculated:
- Degree centrality
- Betweenness centrality
- Closeness centrality
- Eigenvector centrality
- Community structure (Louvain algorithm)

## Troubleshooting

### Memory Issues
```bash
# Reduce sample size
python app.py dashboard --csv file.csv --sample 0.05
```

### Port Already in Use
```bash
# Use different port
python app.py dashboard --csv file.csv --port 8051
```

### Cache Issues
```bash
# Clear cache and reload
python app.py dashboard --csv file.csv --no-cache
```

### Slow Performance
- Start with smaller sample (--sample 0.01)
- Close other applications to free memory
- Use the caching system (don't use --no-cache)
- Consider upgrading RAM

## Development

### Running Individual Components

**Test data processor:**
```python
from data_processor import TraffyDataProcessor

processor = TraffyDataProcessor("bangkok_traffy_30.csv")
df = processor.load_data_chunked(sample_frac=0.1)
stats = processor.get_summary_stats(df)
print(stats)
```

**Generate single map:**
```python
from folium_map import FoliumMapGenerator
from data_processor import TraffyDataProcessor

processor = TraffyDataProcessor("bangkok_traffy_30.csv")
df = processor.load_data_chunked(sample_frac=0.1)

generator = FoliumMapGenerator(df)
map_obj = generator.create_clustered_map(sample_size=5000)
generator.save_map(map_obj, "my_map.html")
```

**Analyze network:**
```python
from network_graph import ComplaintNetworkGraph
from data_processor import TraffyDataProcessor

processor = TraffyDataProcessor("bangkok_traffy_30.csv")
df = processor.load_data_chunked(sample_frac=0.1)

network = ComplaintNetworkGraph(df)
network.build_network(min_edge_weight=5, top_n_nodes=50)
metrics = network.calculate_metrics()
print(metrics)
```

## Technologies Used

- **Dash/Plotly**: Interactive web dashboard and visualizations
- **Folium**: Geospatial map rendering
- **Pandas**: Data manipulation and analysis
- **NetworkX**: Graph analysis
- **Dask**: Parallel computing for large datasets
- **PyArrow**: Efficient data serialization

## Performance Benchmarks

Approximate processing times (on 8GB RAM, i5 processor):

| Operation | 1% Sample | 10% Sample | 100% (900MB) |
|-----------|-----------|------------|--------------|
| First load | ~5s | ~30s | ~5min |
| Cached load | ~1s | ~3s | ~30s |
| Dashboard render | ~2s | ~5s | ~15s |
| Network graph | ~3s | ~10s | ~2min |
| Map generation | ~5s | ~20s | ~5min |

## Contributing

Contributions are welcome! Areas for improvement:
- Additional visualization types
- Real-time data updates
- Machine learning predictions
- Mobile-responsive design
- GeoJSON district boundaries for choropleth
- Multi-language support

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This dashboard is optimized for Bangkok Traffy complaint data but can be adapted for other geospatial datasets with similar structure.
