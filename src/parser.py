from src.cascade import Cascade

def parse_line(line: str) -> Cascade:
    """
    Parses a single line from the raw Weibo dataset into a Cascade object.
    Expected format: id \t root \t start_time \t total_nodes \t paths
    """
    parts = line.strip().split("\t")

    cascade_id = int(parts[0])
    root = parts[1]
    start_time = int(parts[2])

    # Paths represent individual resharing sequences
    paths = parts[4].split()

    edges = []
    timestamps = {root: 0}

    for p in paths:
        # Extract path structure and the specific node's timestamp
        path, t = p.split(":")
        nodes = path.split("/")

        # Record absolute timestamp for the resharing node
        timestamps[nodes[-1]] = int(t)

        # Reconstruct the tree structure by adding edges between parent and child nodes
        for i in range(len(nodes) - 1):
            edges.append((nodes[i], nodes[i + 1]))

    return Cascade(
        cascade_id=cascade_id,
        root=root,
        edges=edges,
        timestamps=timestamps,
        start_time=start_time
    )