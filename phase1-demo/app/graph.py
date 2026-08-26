import networkx as nx
import matplotlib.pyplot as plt
import io
import matplotlib
from .schema import ApartmentState

# Use Agg backend to avoid GUI issues when saving plot to buffer
matplotlib.use('Agg')

def generate_bubble_diagram(state: ApartmentState) -> io.BytesIO:
    G = nx.Graph()
    
    # Add nodes and edges based on state
    for room in state.rooms:
        # Use room type as label, and size if available
        label = f"{room.type}\n({room.size})" if room.size else room.type
        G.add_node(room.id, label=label)
        
        for conn in room.connections:
            # We connect room.id to conn. 
            # Note: conn might be a room type rather than an ID depending on LLM output.
            G.add_edge(room.id, conn)
            
    plt.figure(figsize=(8, 6))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    labels = nx.get_node_attributes(G, 'label')
    # If a node doesn't have a label (e.g., if it was only added via edge), use its ID
    for node in G.nodes():
        if node not in labels:
            labels[node] = str(node)
            
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold', 
            edge_color='gray', arrows=False)
            
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf
