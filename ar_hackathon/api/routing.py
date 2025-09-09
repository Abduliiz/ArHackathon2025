"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name:Phoenix
Email address: abdulrahman.odat@gmail.com
*******************
"""

from typing import Optional
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Determine the next FC to route a package to.
    
    This is the function that students will implement. The game engine will call
    this function for each package at each time step to determine where to route it.
    
    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package
        
    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """
    # If already at destination, stay put
    if package.current_fc == package.destination_fc:
        return None

    # Build adjacency with effective weights and gather outgoing edges from the current FC
    # Effective weight biases away from highly utilized links but keeps true time as the base
    def effective_weight(weight: float, bandwidth: Optional[int], available: Optional[int]) -> float:
        if bandwidth is None or bandwidth <= 0 or available is None:
            return float(weight)
        # Ratio in [0,1]; higher ratio => less penalty
        ratio = max(0.0, min(1.0, available / bandwidth))
        # Up to +50% penalty when link nearly saturated
        multiplier = 1.0 + 0.5 * (1.0 - ratio)
        return float(weight) * multiplier

    # Build forward and reversed adjacency lists for Dijkstra
    adj = {}  # from -> list of (to, eff_w)
    radj = {}  # to -> list of (from, eff_w)  (reversed graph)
    outgoing_from_current = []  # edges: (to, base_w, bandwidth, available, eff_w)
    for conn in state.connections:
        ew = effective_weight(conn.weight, conn.bandwidth, conn.available_bandwidth)
        adj.setdefault(conn.from_fc, []).append((conn.to_fc, ew))
        radj.setdefault(conn.to_fc, []).append((conn.from_fc, ew))
        if conn.from_fc == package.current_fc:
            outgoing_from_current.append(
                (conn.to_fc, conn.weight, conn.bandwidth, conn.available_bandwidth, ew)
            )

    # If no outgoing edges, we cannot move
    if not outgoing_from_current:
        return None

    # Dijkstra on reversed graph from destination to compute dist_to_dest for all nodes
    # This lets us score candidate first hops quickly: cost = eff(current->nbr) + dist_to_dest[nbr]
    import heapq

    INF = float('inf')
    dist_to_dest = {node.id: INF for node in state.fulfillment_centers}
    dest = package.destination_fc
    if dest not in dist_to_dest:
        return None
    dist_to_dest[dest] = 0.0
    pq = [(0.0, dest)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist_to_dest.get(u, INF):
            continue
        for v, w in radj.get(u, []):  # reversed edges
            nd = d + w
            if nd < dist_to_dest.get(v, INF):
                dist_to_dest[v] = nd
                heapq.heappush(pq, (nd, v))

    # Helper to pick best neighbor given a filter on first-hop bandwidth
    def choose_best_neighbor(require_available: bool) -> Optional[str]:
        best_to = None
        best_score = INF
        for to, base_w, bw, avail, eff_w in outgoing_from_current:
            if require_available and bw is not None:
                if avail is None or avail <= 0:
                    continue  # skip saturated first hop

            # Quick win: if direct to destination and (if required) bandwidth available
            if to == dest:
                # Prefer immediate delivery
                return to

            # If neighbor cannot reach destination, skip (unless no info)
            tail = dist_to_dest.get(to, INF)
            if tail == INF:
                continue

            score = eff_w + tail
            if score < best_score:
                best_score = score
                best_to = to
        return best_to

    # First, try requiring available bandwidth on the first hop (Level 3 awareness)
    next_hop = choose_best_neighbor(require_available=True)

    # If none found, relax the requirement (maybe all first hops are currently saturated)
    if next_hop is None:
        next_hop = choose_best_neighbor(require_available=False)

    # As a final fallback, if still none, pick the lightest outgoing edge with any available bandwidth,
    # else the absolute lightest edge; this keeps packages moving in disconnected or temporarily blocked cases
    if next_hop is None:
        # Prefer edges with available bandwidth
        best = None
        best_w = INF
        for to, base_w, bw, avail, _ in outgoing_from_current:
            if bw is not None and (avail is not None) and avail > 0 and base_w < best_w:
                best = to
                best_w = base_w
        if best is None:
            # No available first-hop bandwidth anywhere, just pick the smallest base weight
            for to, base_w, _bw, _avail, _ in outgoing_from_current:
                if base_w < best_w:
                    best = to
                    best_w = base_w
        next_hop = best

    return next_hop
