"""
Main application for Bangkok Traffy Complaints Visualization Dashboard.
Integrates all visualization components.
"""

import argparse
import sys
from pathlib import Path
from data_processor import TraffyDataProcessor
from dashboard import TraffyDashboard
from folium_map import FoliumMapGenerator, generate_all_maps
from network_graph import ComplaintNetworkGraph


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Bangkok Traffy Complaints Visualization Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive dashboard (recommended)
  python app.py dashboard --csv bangkok_traffy.csv --sample 0.1

  # Generate Folium maps
  python app.py maps --csv bangkok_traffy.csv --sample 0.1

  # Generate network graph
  python app.py network --csv bangkok_traffy.csv --sample 0.1

  # Generate all visualizations
  python app.py all --csv bangkok_traffy.csv --sample 0.1

  # Process full dataset (may be slow for 900MB file)
  python app.py dashboard --csv bangkok_traffy.csv --sample 1.0

Notes:
  - For 900MB CSV files, start with --sample 0.1 (10%) for testing
  - Increase sample size gradually based on your system memory
  - The dashboard runs on http://localhost:8050 by default
        """
    )

    parser.add_argument(
        'mode',
        choices=['dashboard', 'maps', 'network', 'all'],
        help='Visualization mode to run'
    )

    parser.add_argument(
        '--csv',
        type=str,
        default='bangkok_traffy_30.csv',
        help='Path to CSV file (default: bangkok_traffy_30.csv)'
    )

    parser.add_argument(
        '--sample',
        type=float,
        default=0.1,
        help='Fraction of data to use (0.0-1.0, default: 0.1 for 10%%)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port for dashboard server (default: 8050)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Ignore cached data and reload from CSV'
    )

    args = parser.parse_args()

    # Validate CSV file
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    # Validate sample fraction
    if not 0.0 < args.sample <= 1.0:
        print("Error: Sample fraction must be between 0.0 and 1.0")
        sys.exit(1)

    print("=" * 70)
    print("BANGKOK TRAFFY COMPLAINTS VISUALIZATION")
    print("=" * 70)
    print(f"CSV File: {args.csv}")
    print(f"Sample Size: {args.sample * 100:.1f}%")
    print(f"Mode: {args.mode}")
    print("=" * 70)
    print()

    # Clear cache if requested
    if args.no_cache:
        import shutil
        cache_dir = Path('cache')
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("Cache cleared.")

    try:
        if args.mode == 'dashboard':
            run_dashboard(args.csv, args.sample, args.port)

        elif args.mode == 'maps':
            run_maps(args.csv, args.sample)

        elif args.mode == 'network':
            run_network(args.csv, args.sample)

        elif args.mode == 'all':
            print("Generating all visualizations...\n")
            run_maps(args.csv, args.sample)
            print()
            run_network(args.csv, args.sample)
            print()
            print("Starting dashboard (press Ctrl+C to stop)...")
            run_dashboard(args.csv, args.sample, args.port)

    except KeyboardInterrupt:
        print("\n\nShutdown requested... exiting")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_dashboard(csv_path: str, sample_frac: float, port: int):
    """Run the interactive Dash dashboard."""
    print("Starting interactive dashboard...\n")
    dashboard = TraffyDashboard(csv_path, sample_frac)
    dashboard.run(debug=False, port=port)


def run_maps(csv_path: str, sample_frac: float):
    """Generate Folium maps."""
    print("Generating Folium maps...\n")
    generate_all_maps(csv_path, sample_frac)
    print("\n✓ Maps generated successfully!")
    print("  Open the HTML files in 'maps/' directory to view them.")


def run_network(csv_path: str, sample_frac: float):
    """Generate network graph."""
    print("Generating complaint type network graph...\n")

    # Load data
    processor = TraffyDataProcessor(csv_path)
    df = processor.load_data_chunked(sample_frac=sample_frac)

    # Create network
    network = ComplaintNetworkGraph(df)
    network.build_network(min_edge_weight=5, top_n_nodes=50)

    # Create visualization
    fig = network.create_plotly_figure(layout_type='spring', show_labels=True)
    fig.write_html("network_graph.html")

    # Save stats
    network.save_network_stats("network_stats.txt")

    print("\n✓ Network graph generated successfully!")
    print("  - Visualization: network_graph.html")
    print("  - Statistics: network_stats.txt")

    # Display top connections
    print("\n" + "=" * 70)
    print("TOP 10 CO-OCCURRING COMPLAINT TYPES")
    print("=" * 70)
    top_connections = network.get_top_connections(10)
    for idx, row in top_connections.iterrows():
        print(f"{idx+1:2d}. {row['type_1']:25s} + {row['type_2']:25s} = {row['co_occurrences']:4d}x")


def quick_stats(csv_path: str):
    """Display quick statistics about the dataset."""
    print("\nLoading data for statistics...")
    processor = TraffyDataProcessor(csv_path)
    df = processor.load_data_chunked(sample_frac=0.01)  # Use 1% sample for quick stats

    stats = processor.get_summary_stats(df)

    print("\n" + "=" * 70)
    print("DATASET STATISTICS (1% sample)")
    print("=" * 70)
    print(f"Total complaints: {stats['total_complaints']:,}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Number of districts: {stats['districts']}")
    print(f"Average solve time: {stats['avg_solve_days']:.1f} days")
    print(f"Reopen rate: {stats['reopen_rate']*100:.1f}%")

    print("\nTop 10 Complaint Types:")
    for i, (ctype, count) in enumerate(list(stats['top_types'].items())[:10], 1):
        print(f"  {i:2d}. {ctype:30s} {count:6,} ({count/stats['total_complaints']*100:5.1f}%)")

    print("\nStatus Distribution:")
    for state, count in stats['state_distribution'].items():
        print(f"  {state:30s} {count:6,} ({count/stats['total_complaints']*100:5.1f}%)")


if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("Bangkok Traffy Complaints Visualization Dashboard\n")
        print("Usage: python app.py [mode] [options]")
        print("\nModes:")
        print("  dashboard  - Run interactive web dashboard")
        print("  maps       - Generate static Folium maps")
        print("  network    - Generate complaint type network graph")
        print("  all        - Generate all visualizations")
        print("\nFor detailed help, run: python app.py --help")
        print("\nQuick start:")
        print("  python app.py dashboard --csv bangkok_traffy_30.csv --sample 0.1")
        sys.exit(0)

    main()
