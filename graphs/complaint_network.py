"""
Graph Network Visualization for Complaint Relationships
Analyzes co-occurrence patterns, organization networks, and community detection
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from networkx.algorithms import community as nx_community
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplaintNetworkAnalyzer:
    """Network graph analysis for urban complaints"""

    def __init__(self):
        self.G = nx.Graph()
        self.complaint_network = None
        self.organization_network = None

    def build_complaint_type_network(self, df: pd.DataFrame):
        """
        Build network based on co-occurrence of complaint types
        Edge weight = number of times two types appear together
        """
        logger.info("Building complaint type co-occurrence network...")

        # Parse multi-label complaint types
        def parse_types(type_str):
            if pd.isna(type_str):
                return []
            # Remove curly braces and split
            cleaned = str(type_str).replace('{', '').replace('}', '')
            return [t.strip() for t in cleaned.split(',') if t.strip()]

        df['types_list'] = df['type'].apply(parse_types)

        # Count co-occurrences
        co_occurrence = Counter()

        for types in df['types_list']:
            if len(types) > 1:
                # All pairs of types in same complaint
                for pair in itertools.combinations(sorted(types), 2):
                    co_occurrence[pair] += 1

        # Build graph
        self.complaint_network = nx.Graph()

        for (type1, type2), weight in co_occurrence.items():
            if weight >= 5:  # Minimum threshold
                self.complaint_network.add_edge(type1, type2, weight=weight)

        # Add isolated nodes for completeness
        all_types = set()
        for types in df['types_list']:
            all_types.update(types)

        for t in all_types:
            if t not in self.complaint_network:
                self.complaint_network.add_node(t)

        logger.info(f"Created network with {self.complaint_network.number_of_nodes()} nodes "
                   f"and {self.complaint_network.number_of_edges()} edges")

        return self.complaint_network

    def build_organization_network(self, df: pd.DataFrame):
        """
        Build network of organizations handling similar complaints
        """
        logger.info("Building organization collaboration network...")

        # Create edges between organizations handling same complaint type
        type_orgs = df.groupby('type')['organization'].apply(list).to_dict()

        self.organization_network = nx.Graph()

        for complaint_type, orgs in type_orgs.items():
            orgs_unique = list(set([str(o) for o in orgs if pd.notna(o)]))

            if len(orgs_unique) > 1:
                for org1, org2 in itertools.combinations(orgs_unique, 2):
                    if self.organization_network.has_edge(org1, org2):
                        self.organization_network[org1][org2]['weight'] += 1
                    else:
                        self.organization_network.add_edge(org1, org2, weight=1)

        logger.info(f"Created org network with {self.organization_network.number_of_nodes()} nodes "
                   f"and {self.organization_network.number_of_edges()} edges")

        return self.organization_network

    def detect_communities(self, G: nx.Graph):
        """Detect communities using greedy modularity algorithm"""
        logger.info("Detecting communities...")

        if len(G.nodes()) == 0:
            return {}

        # Use greedy modularity communities from NetworkX
        communities_generator = nx_community.greedy_modularity_communities(G, weight='weight')
        communities_list = list(communities_generator)

        # Convert to dict format {node: community_id}
        partition = {}
        for comm_id, comm_nodes in enumerate(communities_list):
            for node in comm_nodes:
                partition[node] = comm_id

        n_communities = len(communities_list)
        logger.info(f"Detected {n_communities} communities")

        return partition

    def calculate_centrality_metrics(self, G: nx.Graph):
        """Calculate various centrality metrics"""
        logger.info("Calculating centrality metrics...")

        metrics = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G, weight='weight'),
            'closeness_centrality': nx.closeness_centrality(G, distance='weight'),
            'eigenvector_centrality': nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
                                     if len(G.nodes()) > 0 else {}
        }

        return metrics

    def visualize_network_matplotlib(self, G: nx.Graph, title: str = "Complaint Network",
                                    communities: dict = None):
        """Visualize network using matplotlib"""
        plt.figure(figsize=(16, 12))

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50, weight='weight')

        # Node colors based on communities
        if communities:
            node_colors = [communities.get(node, 0) for node in G.nodes()]
            n_communities = len(set(communities.values()))
            cmap = plt.cm.get_cmap('tab20', n_communities)
        else:
            node_colors = 'lightblue'
            cmap = None

        # Node sizes based on degree
        node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]

        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=cmap,
            alpha=0.7,
            edgecolors='black',
            linewidths=1
        )

        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v].get('weight', 1) for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(
            G, pos,
            width=[w/max_weight * 3 for w in weights],
            alpha=0.3,
            edge_color='gray'
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold'
        )

        plt.title(title, fontsize=18, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Save
        output_path = f"visualization/graphs/outputs/{title.lower().replace(' ', '_')}.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Network visualization saved to {output_path}")

        return plt.gcf()

    def visualize_network_plotly(self, G: nx.Graph, title: str = "Complaint Network",
                                communities: dict = None):
        """Create interactive network visualization with Plotly"""
        logger.info("Creating interactive Plotly visualization...")

        if len(G.nodes()) == 0:
            logger.warning("Empty graph, skipping visualization")
            return None

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50, weight='weight')

        # Edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]].get('weight', 1)

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight/10, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node info
            degree = G.degree(node)
            node_text.append(f"{node}<br>Degree: {degree}")
            node_size.append(20 + degree * 5)

            if communities:
                node_color.append(communities.get(node, 0))
            else:
                node_color.append(degree)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(n) for n in G.nodes()],
            hovertext=node_text,
            textposition='top center',
            textfont=dict(size=10),
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Community' if communities else 'Degree',
                    thickness=15,
                    xanchor='left'
                ),
                line=dict(width=2, color='white')
            )
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=800
        )

        # Save
        output_path = f"visualization/graphs/outputs/{title.lower().replace(' ', '_')}_interactive.html"
        fig.write_html(output_path)
        logger.info(f"Interactive visualization saved to {output_path}")

        return fig

    def analyze_network_properties(self, G: nx.Graph):
        """Analyze graph properties and statistics"""
        logger.info("Analyzing network properties...")

        if len(G.nodes()) == 0:
            return {}

        properties = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
        }

        if nx.is_connected(G):
            properties['diameter'] = nx.diameter(G)
            properties['avg_shortest_path'] = nx.average_shortest_path_length(G)

        properties['avg_clustering'] = nx.average_clustering(G)
        properties['transitivity'] = nx.transitivity(G)

        # Degree statistics
        degrees = [d for n, d in G.degree()]
        properties['avg_degree'] = np.mean(degrees)
        properties['max_degree'] = max(degrees) if degrees else 0
        properties['min_degree'] = min(degrees) if degrees else 0

        logger.info("\n" + "=" * 60)
        logger.info("NETWORK PROPERTIES")
        logger.info("=" * 60)
        for key, value in properties.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 60)

        return properties

    def export_graph(self, G: nx.Graph, filename: str):
        """Export graph to various formats"""
        output_dir = Path("visualization/graphs/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # GraphML format
        nx.write_graphml(G, str(output_dir / f"{filename}.graphml"))

        # GML format
        nx.write_gml(G, str(output_dir / f"{filename}.gml"))

        # Edge list
        nx.write_edgelist(G, str(output_dir / f"{filename}_edges.txt"))

        logger.info(f"Exported graph to {output_dir}/{filename}.*")


def main():
    """Main network analysis pipeline"""
    logger.info("=" * 80)
    logger.info("Complaint Network Analysis")
    logger.info("=" * 80)

    # Load data (simulated)
    np.random.seed(42)
    n_samples = 5000

    complaint_types = ['น้ำท่วม', 'จราจร', 'ความสะอาด', 'ถนน', 'ทางเท้า',
                      'สะพาน', 'ท่อระบายน้ำ', 'ไฟฟ้า']

    organizations = [
        'เขตปทุมวัน', 'เขตห้วยขวาง', 'สำนักการระบายน้ำ',
        'สำนักการจราจร', 'การไฟฟ้านครหลวง', 'เขตดินแดง'
    ]

    # Simulate multi-label types
    types_list = []
    for _ in range(n_samples):
        n_types = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        selected_types = np.random.choice(complaint_types, size=n_types, replace=False)
        types_list.append('{' + ','.join(selected_types) + '}')

    df = pd.DataFrame({
        'type': types_list,
        'organization': np.random.choice(organizations, n_samples)
    })

    # Initialize analyzer
    analyzer = ComplaintNetworkAnalyzer()

    # 1. Build complaint type network
    complaint_graph = analyzer.build_complaint_type_network(df)

    # 2. Detect communities
    communities = analyzer.detect_communities(complaint_graph)

    # 3. Calculate centrality
    centrality = analyzer.calculate_centrality_metrics(complaint_graph)

    logger.info("\nTop 5 nodes by degree centrality:")
    sorted_centrality = sorted(centrality['degree_centrality'].items(),
                              key=lambda x: x[1], reverse=True)
    for node, score in sorted_centrality[:5]:
        logger.info(f"  {node}: {score:.3f}")

    # 4. Visualize
    analyzer.visualize_network_matplotlib(complaint_graph,
                                         title="Complaint Type Co-occurrence Network",
                                         communities=communities)

    analyzer.visualize_network_plotly(complaint_graph,
                                      title="Interactive Complaint Network",
                                      communities=communities)

    # 5. Analyze properties
    properties = analyzer.analyze_network_properties(complaint_graph)

    # 6. Build organization network
    org_graph = analyzer.build_organization_network(df)
    org_communities = analyzer.detect_communities(org_graph)

    analyzer.visualize_network_matplotlib(org_graph,
                                         title="Organization Collaboration Network",
                                         communities=org_communities)

    # 7. Export graphs
    analyzer.export_graph(complaint_graph, "complaint_network")
    analyzer.export_graph(org_graph, "organization_network")

    logger.info("=" * 80)
    logger.info("Network analysis completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()