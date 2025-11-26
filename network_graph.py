"""
Network graph visualization for Bangkok Traffy complaint types.
Shows relationships and co-occurrences between different complaint types.
"""

import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
from itertools import combinations
import community as community_louvain
from data_processor import TraffyDataProcessor


class ComplaintNetworkGraph:
    """Generate network graphs showing complaint type relationships."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize network graph generator.

        Args:
            df: DataFrame with complaint data including complaint_types
        """
        self.df = df
        self.graph = None

    def build_network(self, min_edge_weight: int = 5, top_n_nodes: int = 50):
        """
        Build network graph from complaint co-occurrences.

        Args:
            min_edge_weight: Minimum co-occurrence count for edge
            top_n_nodes: Maximum number of nodes to include

        Returns:
            NetworkX graph object
        """
        print("Building complaint type network...")

        # Count co-occurrences
        edge_counts = Counter()
        node_counts = Counter()

        for types in self.df['complaint_types']:
            if len(types) > 0:
                # Count individual types
                for t in types:
                    node_counts[t] += 1

                # Count pairs (co-occurrences)
                if len(types) > 1:
                    for pair in combinations(sorted(types), 2):
                        edge_counts[pair] += 1

        # Get top nodes
        top_types = [t for t, _ in node_counts.most_common(top_n_nodes)]
        print(f"  Top {len(top_types)} complaint types selected")

        # Create graph
        G = nx.Graph()

        # Add nodes with attributes
        for node_type in top_types:
            G.add_node(
                node_type,
                count=node_counts[node_type],
                label=node_type
            )

        # Add edges
        edge_count = 0
        for (source, target), weight in edge_counts.items():
            if (source in top_types and
                target in top_types and
                weight >= min_edge_weight):
                G.add_edge(source, target, weight=weight)
                edge_count += 1

        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        self.graph = G
        return G

    def detect_communities(self):
        """Detect communities in the network using Louvain algorithm."""
        if self.graph is None:
            raise ValueError("Graph not built. Call build_network() first.")

        # Detect communities
        communities = community_louvain.best_partition(self.graph)

        # Add community attribute to nodes
        nx.set_node_attributes(self.graph, communities, 'community')

        print(f"  Detected {len(set(communities.values()))} communities")

        return communities

    def calculate_metrics(self):
        """Calculate network metrics."""
        if self.graph is None:
            raise ValueError("Graph not built. Call build_network() first.")

        metrics = {
            'degree_centrality': nx.degree_centrality(self.graph),
            'betweenness_centrality': nx.betweenness_centrality(self.graph),
            'closeness_centrality': nx.closeness_centrality(self.graph),
            'eigenvector_centrality': nx.eigenvector_centrality(self.graph, max_iter=1000)
        }

        return metrics

    def create_plotly_figure(self, layout_type: str = 'spring', show_labels: bool = True):
        """
        Create interactive Plotly figure of the network.

        Args:
            layout_type: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            show_labels: Whether to show node labels

        Returns:
            Plotly figure object
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_network() first.")

        print("Creating network visualization...")

        # Detect communities
        communities = self.detect_communities()

        # Calculate layout
        if layout_type == 'spring':
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        elif layout_type == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout_type == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Create edge trace
        edge_trace = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = self.graph[edge[0]][edge[1]]['weight']

            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=0.5 + (weight / 50),  # Scale line width by weight
                        color='rgba(125, 125, 125, 0.3)'
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
            )

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node info
            count = self.graph.nodes[node]['count']
            community = communities[node]
            degree = self.graph.degree[node]

            node_text.append(
                f"<b>{node}</b><br>" +
                f"Count: {count:,}<br>" +
                f"Connections: {degree}<br>" +
                f"Community: {community}"
            )

            node_color.append(community)
            node_size.append(10 + (count / 100))  # Scale by count

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text' if show_labels else 'markers',
            text=[self.graph.nodes[node]['label'] for node in self.graph.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Community",
                    thickness=15,
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=1, color='white')
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])

        fig.update_layout(
            title={
                'text': "Bangkok Traffy Complaint Type Network<br>" +
                        "<sub>Node size = complaint frequency | Edges = co-occurrences | Colors = communities</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=800
        )

        return fig

    def get_top_connections(self, n: int = 10):
        """
        Get top connected complaint type pairs.

        Args:
            n: Number of top pairs to return

        Returns:
            DataFrame with top connections
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_network() first.")

        # Get edges sorted by weight
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'type_1': u,
                'type_2': v,
                'co_occurrences': data['weight']
            })

        df_edges = pd.DataFrame(edges)
        df_edges = df_edges.sort_values('co_occurrences', ascending=False).head(n)

        return df_edges

    def get_central_nodes(self, n: int = 10):
        """
        Get most central nodes by different metrics.

        Args:
            n: Number of top nodes to return

        Returns:
            Dictionary of DataFrames for each centrality metric
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_network() first.")

        metrics = self.calculate_metrics()

        results = {}
        for metric_name, metric_values in metrics.items():
            sorted_nodes = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)[:n]
            results[metric_name] = pd.DataFrame(sorted_nodes, columns=['complaint_type', metric_name])

        return results

    def save_network_stats(self, output_file: str = "network_stats.txt"):
        """Save network statistics to file."""
        if self.graph is None:
            raise ValueError("Graph not built. Call build_network() first.")

        metrics = self.calculate_metrics()
        communities = community_louvain.best_partition(self.graph)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BANGKOK TRAFFY COMPLAINT TYPE NETWORK ANALYSIS\n")
            f.write("=" * 60 + "\n\n")

            # Basic stats
            f.write("BASIC STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Number of nodes (complaint types): {self.graph.number_of_nodes()}\n")
            f.write(f"Number of edges (co-occurrences): {self.graph.number_of_edges()}\n")
            f.write(f"Network density: {nx.density(self.graph):.4f}\n")
            f.write(f"Average degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}\n")
            f.write(f"Number of communities: {len(set(communities.values()))}\n\n")

            # Top nodes by degree
            f.write("TOP 10 MOST CONNECTED COMPLAINT TYPES\n")
            f.write("-" * 60 + "\n")
            degree_sorted = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:10]
            for i, (node, degree) in enumerate(degree_sorted, 1):
                f.write(f"{i:2d}. {node:30s} - {degree} connections\n")
            f.write("\n")

            # Top edges
            f.write("TOP 10 MOST FREQUENT CO-OCCURRENCES\n")
            f.write("-" * 60 + "\n")
            edges_sorted = sorted(self.graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
            for i, (u, v, data) in enumerate(edges_sorted, 1):
                f.write(f"{i:2d}. {u} + {v}: {data['weight']} times\n")
            f.write("\n")

            # Community distribution
            f.write("COMMUNITY DISTRIBUTION\n")
            f.write("-" * 60 + "\n")
            community_sizes = Counter(communities.values())
            for comm_id, size in sorted(community_sizes.items()):
                f.write(f"Community {comm_id}: {size} complaint types\n")
                # List types in this community
                types_in_comm = [node for node, comm in communities.items() if comm == comm_id]
                f.write(f"  Types: {', '.join(types_in_comm[:5])}")
                if len(types_in_comm) > 5:
                    f.write(f" ... and {len(types_in_comm) - 5} more")
                f.write("\n\n")

        print(f"Network statistics saved to: {output_file}")


def create_network_dashboard(csv_path: str, sample_frac: float = 0.1):
    """
    Create network graph visualization.

    Args:
        csv_path: Path to CSV file
        sample_frac: Fraction of data to use
    """
    # Load data
    processor = TraffyDataProcessor(csv_path)
    df = processor.load_data_chunked(sample_frac=sample_frac)

    # Create network
    network = ComplaintNetworkGraph(df)
    network.build_network(min_edge_weight=5, top_n_nodes=50)

    # Create visualization
    fig = network.create_plotly_figure(layout_type='spring', show_labels=True)

    # Save as HTML
    fig.write_html("network_graph.html")
    print("Network graph saved to: network_graph.html")

    # Save stats
    network.save_network_stats("network_stats.txt")

    # Print top connections
    print("\nTop 10 Co-occurring Complaint Types:")
    print(network.get_top_connections(10))

    return network


if __name__ == "__main__":
    # Create network visualization
    network = create_network_dashboard("bangkok_traffy_30.csv", sample_frac=0.1)
