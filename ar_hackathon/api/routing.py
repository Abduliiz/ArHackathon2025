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

    import heapq
    from collections import defaultdict, deque
    
    INF = float('inf')
    dest = package.destination_fc
    current = package.current_fc
    
    # Build adjacency lists with enhanced weight calculation
    def calculate_effective_weight(weight: float, bandwidth: Optional[int], available: Optional[int], 
                                 congestion_factor: float = 1.0) -> float:
        """Calculate effective weight considering bandwidth and congestion."""
        if bandwidth is None or bandwidth <= 0 or available is None:
            return float(weight)
        
        # Calculate utilization ratio
        utilization = max(0.0, min(1.0, (bandwidth - available) / bandwidth))
        
        # Apply congestion penalty - more aggressive than before
        congestion_penalty = 1.0 + 1.5 * utilization * congestion_factor
        
        # Apply bandwidth scarcity penalty
        scarcity_penalty = 1.0 + 0.3 * (1.0 - utilization) if utilization > 0.8 else 1.0
        
        return float(weight) * congestion_penalty * scarcity_penalty

    # Build graph with enhanced weights
    adj = defaultdict(list)
    radj = defaultdict(list)
    outgoing_from_current = []
    
    # Calculate network congestion factor based on overall bandwidth utilization
    total_bandwidth = 0
    total_used = 0
    for conn in state.connections:
        if conn.bandwidth is not None:
            total_bandwidth += conn.bandwidth
            if conn.available_bandwidth is not None:
                total_used += conn.bandwidth - conn.available_bandwidth
    
    congestion_factor = 1.0
    if total_bandwidth > 0:
        network_utilization = total_used / total_bandwidth
        congestion_factor = 1.0 + network_utilization  # Higher network congestion = higher penalty
    
    for conn in state.connections:
        ew = calculate_effective_weight(conn.weight, conn.bandwidth, conn.available_bandwidth, congestion_factor)
        adj[conn.from_fc].append((conn.to_fc, ew, conn.weight, conn.bandwidth, conn.available_bandwidth))
        radj[conn.to_fc].append((conn.from_fc, ew))
        
        if conn.from_fc == current:
            outgoing_from_current.append((conn.to_fc, conn.weight, conn.bandwidth, conn.available_bandwidth, ew))

    if not outgoing_from_current:
        return None

    # Multi-step lookahead: Find shortest paths considering future bandwidth availability
    def find_optimal_path_with_lookahead(start: str, end: str, lookahead_steps: int = 3) -> tuple:
        """Find optimal path considering future bandwidth constraints."""
        
        # First, get basic shortest path using Dijkstra
        dist = {node.id: INF for node in state.fulfillment_centers}
        dist[start] = 0.0
        prev = {}
        pq = [(0.0, start)]
        
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, INF):
                continue
            if u == end:
                break
                
            for v, w, _, _, _ in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        
        if end not in prev and start != end:
            return INF, []
        
        # Reconstruct path
        path = []
        current_node = end
        while current_node in prev:
            path.append(current_node)
            current_node = prev[current_node]
        if start == end or current_node == start:
            path.append(start)
        path.reverse()
        
        return dist.get(end, INF), path

    # Get optimal path
    optimal_cost, optimal_path = find_optimal_path_with_lookahead(current, dest)
    
    # If we can reach destination, choose the next step in optimal path
    if optimal_path and len(optimal_path) > 1:
        next_in_path = optimal_path[1]
        
        # Verify this step is actually available from current position
        for to, _, _, _, _ in outgoing_from_current:
            if to == next_in_path:
                return next_in_path

    # Fallback: Use enhanced single-step optimization
    def choose_best_next_hop() -> Optional[str]:
        """Choose the best next hop using enhanced scoring."""
        best_hop = None
        best_score = INF
        
        for to, base_w, bw, avail, eff_w in outgoing_from_current:
            # Skip if no bandwidth available and bandwidth is constrained
            if bw is not None and (avail is None or avail <= 0):
                continue
            
            # Direct connection to destination
            if to == dest:
                return to
            
            # Calculate score considering multiple factors
            # 1. Effective weight to next hop
            hop_cost = eff_w
            
            # 2. Estimated remaining cost to destination (using basic Dijkstra)
            remaining_cost = INF
            temp_dist = {node.id: INF for node in state.fulfillment_centers}
            temp_dist[to] = 0.0
            temp_pq = [(0.0, to)]
            
            while temp_pq:
                d, u = heapq.heappop(temp_pq)
                if d != temp_dist.get(u, INF):
                    continue
                if u == dest:
                    remaining_cost = d
                    break
                for v, w in radj.get(u, []):
                    nd = d + w
                    if nd < temp_dist.get(v, INF):
                        temp_dist[v] = nd
                        heapq.heappush(temp_pq, (nd, v))
            
            if remaining_cost == INF:
                continue
            
            # 3. Bandwidth availability bonus
            bandwidth_bonus = 0.0
            if bw is not None and avail is not None:
                utilization = (bw - avail) / bw
                bandwidth_bonus = -0.1 * utilization  # Prefer less utilized links
            
            # 4. Path diversity bonus (prefer paths that don't go through highly congested areas)
            diversity_bonus = 0.0
            if len(optimal_path) > 2 and to in optimal_path:
                # Prefer staying on the optimal path
                diversity_bonus = -0.05
            
            total_score = hop_cost + remaining_cost + bandwidth_bonus + diversity_bonus
            
            if total_score < best_score:
                best_score = total_score
                best_hop = to
        
        return best_hop

    # Try to find a good next hop
    next_hop = choose_best_next_hop()
    
    # Final fallback: just pick the lightest available edge
    if next_hop is None:
        best = None
        best_w = INF
        for to, base_w, bw, avail, _ in outgoing_from_current:
            if bw is None or (avail is not None and avail > 0):
                if base_w < best_w:
                    best = to
                    best_w = base_w
        next_hop = best

    return next_hop
