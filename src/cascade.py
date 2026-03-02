import networkx as nx

class Cascade:
    def __init__(self, cascade_id, root, edges, timestamps, start_time):
        self.cascade_id = cascade_id
        self.root = root
        self.timestamps = timestamps  # node_id: absolute_timestamp
        self.start_time = start_time

        # Initialize directed graph for the cascade
        self.graph = nx.DiGraph()
        self.graph.add_node(root)
        for u, v in edges:
            self.graph.add_edge(u, v)

    # Structural features

    def size(self):
        """Returns the number of nodes in the cascade."""
        return self.graph.number_of_nodes()

    def depth(self):
        """Returns the maximum distance from the root."""
        lengths = nx.single_source_shortest_path_length(self.graph, self.root)
        return max(lengths.values()) if lengths else 0

    def breadth(self):
        """Returns the maximum number of nodes at any single level."""
        levels = {}
        lengths = nx.single_source_shortest_path_length(self.graph, self.root)
        for d in lengths.values():
            levels[d] = levels.get(d, 0) + 1
        return max(levels.values()) if levels else 0

    def wiener_index(self):
        """
        Computes the average distance between all node pairs.
        Represents structural virality.
        """
        n = self.graph.number_of_nodes()
        if n < 2:
            return 0.0
        
        undirected_g = self.graph.to_undirected()
        path_lengths = dict(nx.all_pairs_shortest_path_length(undirected_g))
        
        total_distance = 0
        pairs_count = 0
        for node_u in path_lengths:
            for node_v, dist in path_lengths[node_u].items():
                if node_u != node_v:
                    total_distance += dist
                    pairs_count += 1
        
        return total_distance / pairs_count if pairs_count > 0 else 0.0

    # Temporal features

    def times_relative(self):
        """Returns sorted timestamps relative to the start time."""
        return sorted([t - self.start_time for t in self.timestamps.values()])

    def duration(self):
        """Returns the time elapsed since the start of the cascade."""
        t_rel = self.times_relative()
        return max(t_rel) if t_rel else 0

    def acceleration(self):
        """
        Compares average speed of the first half vs the second half.
        Positive value indicates acceleration.
        """
        t_rel = self.times_relative()
        k = len(t_rel)
        if k < 4:
            return 0.0
        
        mid = k // 2
        first_half_duration = t_rel[mid-1] - t_rel[0]
        second_half_duration = t_rel[-1] - t_rel[mid-1]
        
        if first_half_duration == 0 or second_half_duration == 0:
            return 0.0
        
        v1 = mid / first_half_duration
        v2 = (k - mid) / second_half_duration
        
        return v2 - v1

    # Partial observation

    def subcascade(self, k):
        """Returns a new Cascade object containing only the first k events."""
        if k >= len(self.timestamps):
            return self

        # Sort nodes by time and select the first k
        observed_items = sorted(self.timestamps.items(), key=lambda x: x[1])[:k]
        observed_nodes = {n for n, _ in observed_items}
        
        # Filter edges and timestamps for the observed subset
        edges = [(u, v) for u, v in self.graph.edges() if u in observed_nodes and v in observed_nodes]
        timestamps = {n: self.timestamps[n] for n in observed_nodes}

        return Cascade(
            cascade_id=self.cascade_id,
            root=self.root,
            edges=edges,
            timestamps=timestamps,
            start_time=self.start_time
        )
