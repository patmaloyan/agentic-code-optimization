"""
Initial program for shortest path optimization example.

This program computes single-source shortest paths in weighted graphs.
The goal is to evolve a more efficient shortest-path implementation.

The initial implementation uses a simple adjacency-matrix version of
Dijkstra's algorithm. This is correct, but inefficient for sparse graphs.

Initial complexity:
    Time:  O(V^2 + E)
    Space: O(V^2)

A better implementation should notice that the input is given as an edge list
and can be converted to an adjacency list, then processed with a priority queue.
"""

import random
import math
import heapq


INF = float("inf")


def generate_weighted_graph(num_nodes, regime, seed):
    """
    Generate a connected undirected weighted graph.

    Args:
        num_nodes: number of vertices
        regime: one of "sparse", "medium", "dense"
        seed: random seed for reproducibility

    Returns:
        edges: list of (u, v, weight)
    """
    rng = random.Random(seed)

    if regime == "sparse":
        target_edges = 3 * num_nodes
    elif regime == "medium":
        target_edges = int(num_nodes * math.log2(num_nodes))
    elif regime == "dense":
        target_edges = (num_nodes * num_nodes) // 4
    else:
        raise ValueError(f"Unknown regime: {regime}")

    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    target_edges = min(target_edges, max_possible_edges)

    edges = []
    used = set()

    # First build a random spanning tree so the graph is connected.
    for v in range(1, num_nodes):
        u = rng.randint(0, v - 1)
        weight = rng.randint(1, 20)

        a, b = min(u, v), max(u, v)
        used.add((a, b))
        edges.append((a, b, weight))

    # Add extra random edges until reaching the target edge count.
    while len(edges) < target_edges:
        u = rng.randint(0, num_nodes - 1)
        v = rng.randint(0, num_nodes - 1)

        if u == v:
            continue

        a, b = min(u, v), max(u, v)

        if (a, b) in used:
            continue

        used.add((a, b))
        weight = rng.randint(1, 20)
        edges.append((a, b, weight))

    return edges


# EVOLVE-BLOCK-START
def shortest_paths(num_nodes, edges, source):
    """
    Compute shortest path distances from source to all nodes.

    Current implementation: adjacency-matrix Dijkstra.

    This is intentionally inefficient for sparse graphs because it stores
    all possible edges in a V x V matrix and scans all nodes repeatedly.
    """
    graph = [[INF] * num_nodes for _ in range(num_nodes)]

    for i in range(num_nodes):
        graph[i][i] = 0

    for u, v, weight in edges:
        graph[u][v] = min(graph[u][v], weight)
        graph[v][u] = min(graph[v][u], weight)

    distances = [INF] * num_nodes
    visited = [False] * num_nodes
    distances[source] = 0

    for _ in range(num_nodes):
        current = -1
        current_distance = INF

        # Find the unvisited node with the smallest known distance.
        for node in range(num_nodes):
            if not visited[node] and distances[node] < current_distance:
                current_distance = distances[node]
                current = node

        if current == -1:
            break

        visited[current] = True

        # Scan every possible neighbor.
        for neighbor in range(num_nodes):
            weight = graph[current][neighbor]

            if weight == INF or visited[neighbor]:
                continue

            new_distance = distances[current] + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance

    return distances
# EVOLVE-BLOCK-END


def reference_shortest_paths(num_nodes, edges, source):
    """
    Reference implementation using adjacency-list Dijkstra.

    This is outside the EVOLVE-BLOCK and is used only for correctness checking.
    """
    graph = [[] for _ in range(num_nodes)]

    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))

    distances = [INF] * num_nodes
    distances[source] = 0

    heap = [(0, source)]

    while heap:
        current_distance, node = heapq.heappop(heap)

        if current_distance != distances[node]:
            continue

        for neighbor, weight in graph[node]:
            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))

    return distances


def verify_distances(actual, expected):
    """Verify that two distance arrays match."""
    if len(actual) != len(expected):
        return False

    for a, b in zip(actual, expected):
        if a == INF and b == INF:
            continue

        if a != b:
            return False

    return True


def main():
    """Run shortest path algorithm on several graph sizes and regimes."""
    import time

    regimes = ["sparse", "medium", "dense"]
    sizes = [50, 200, 500, 1000]

    total_time = 0.0

    for regime in regimes:
        for size in sizes:
            seed = 1000 + size + 17 * len(regime)
            edges = generate_weighted_graph(size, regime, seed)
            source = 0

            expected = reference_shortest_paths(size, edges, source)

            start_time = time.perf_counter()
            actual = shortest_paths(size, edges, source)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            total_time += elapsed

            if not verify_distances(actual, expected):
                print(f"ERROR: Incorrect distances for regime {regime}, size {size}")
                return None

            print(
                f"Regime {regime:6s} Size {size:4d} "
                f"Edges {len(edges):6d}: {elapsed * 1000:.3f} ms"
            )

    print(f"Total time: {total_time * 1000:.3f} ms")
    return total_time


if __name__ == "__main__":
    main()