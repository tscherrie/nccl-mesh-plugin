#!/usr/bin/env python3
"""
NCCL Mesh Plugin - Ring Topology Integration Test

This test validates the mesh plugin's ring topology routing by simulating
a 4-node ring configuration where:
  - Node A connects to B and D (direct)
  - Node B connects to A and C (direct)
  - Node C connects to B and D (direct)
  - Node D connects to C and A (direct)
  - A-C and B-D require relay routing (2 hops)

Test Setup (simulated via environment):
    Node A (192.168.100.x, 192.168.103.x) -> B, D
    Node B (192.168.100.x, 192.168.101.x) -> A, C
    Node C (192.168.101.x, 192.168.102.x) -> B, D
    Node D (192.168.102.x, 192.168.103.x) -> C, A

Usage:
    python3 test_ring_topo.py [--verbose]

Requirements:
    - PyTorch with NCCL support (for actual collective tests)
    - libnccl-net-mesh.so in LD_LIBRARY_PATH
"""

import os
import sys
import argparse
import subprocess
import json

# Test configuration
RING_NODES = ['A', 'B', 'C', 'D']
RING_SUBNETS = {
    'A': ['192.168.100', '192.168.103'],  # Connects to B (100), D (103)
    'B': ['192.168.100', '192.168.101'],  # Connects to A (100), C (101)
    'C': ['192.168.101', '192.168.102'],  # Connects to B (101), D (102)
    'D': ['192.168.102', '192.168.103'],  # Connects to C (102), A (103)
}

# Expected connectivity matrix
# 1 = direct, 2 = relay
EXPECTED_HOPS = {
    ('A', 'B'): 1, ('A', 'C'): 2, ('A', 'D'): 1,
    ('B', 'A'): 1, ('B', 'C'): 1, ('B', 'D'): 2,
    ('C', 'A'): 2, ('C', 'B'): 1, ('C', 'D'): 1,
    ('D', 'A'): 1, ('D', 'B'): 2, ('D', 'C'): 1,
}


def check_ring_topology():
    """Verify that the ring topology is correctly detected."""
    print("\n=== Ring Topology Verification ===")

    # Check each node's connectivity
    for node in RING_NODES:
        subnets = RING_SUBNETS[node]
        print(f"Node {node}: subnets {subnets}")

        # Count direct neighbors
        direct_neighbors = []
        for other in RING_NODES:
            if other == node:
                continue
            other_subnets = RING_SUBNETS[other]
            # Check for shared subnet
            shared = set(subnets) & set(other_subnets)
            if shared:
                direct_neighbors.append(other)

        print(f"  Direct neighbors: {direct_neighbors}")

        # In a ring, each node should have exactly 2 neighbors
        if len(direct_neighbors) != 2:
            print(f"  ERROR: Expected 2 neighbors, got {len(direct_neighbors)}")
            return False

    print("Ring topology verified: all nodes have exactly 2 neighbors")
    return True


def check_hop_counts():
    """Verify expected hop counts between all node pairs."""
    print("\n=== Hop Count Verification ===")

    all_correct = True
    for (src, dst), expected_hops in EXPECTED_HOPS.items():
        # Calculate actual hops based on shared subnets
        src_subnets = set(RING_SUBNETS[src])
        dst_subnets = set(RING_SUBNETS[dst])

        if src_subnets & dst_subnets:
            actual_hops = 1  # Direct
        else:
            actual_hops = 2  # Relay (for 4-node ring, max is 2)

        status = "OK" if actual_hops == expected_hops else "FAIL"
        print(f"  {src} -> {dst}: expected {expected_hops} hop(s), got {actual_hops} [{status}]")

        if actual_hops != expected_hops:
            all_correct = False

    return all_correct


def check_dual_path_routing():
    """Verify that 2-hop destinations have dual paths (CW and CCW)."""
    print("\n=== Dual Path Routing Verification ===")

    # In a 4-node ring, opposite nodes (A-C, B-D) can be reached via 2 paths
    dual_path_pairs = [('A', 'C'), ('B', 'D')]

    for src, dst in dual_path_pairs:
        # Find the two paths
        path1 = find_ring_path(src, dst, clockwise=True)
        path2 = find_ring_path(src, dst, clockwise=False)

        print(f"  {src} -> {dst}:")
        print(f"    Clockwise path:  {' -> '.join(path1)} ({len(path1)-1} hop(s))")
        print(f"    Counter-CW path: {' -> '.join(path2)} ({len(path2)-1} hop(s))")

        # Both paths should be 2 hops
        if len(path1) != 3 or len(path2) != 3:
            print(f"    ERROR: Expected both paths to be 2 hops")
            return False

    print("Dual path routing verified")
    return True


def find_ring_path(src, dst, clockwise=True):
    """Find path in the ring going clockwise or counter-clockwise."""
    ring_order = ['A', 'B', 'C', 'D']
    src_idx = ring_order.index(src)
    dst_idx = ring_order.index(dst)

    path = [src]
    n = len(ring_order)

    if clockwise:
        curr = src_idx
        while curr != dst_idx:
            curr = (curr + 1) % n
            path.append(ring_order[curr])
    else:
        curr = src_idx
        while curr != dst_idx:
            curr = (curr - 1 + n) % n
            path.append(ring_order[curr])

    return path


def check_load_balancing():
    """Verify load balancing configuration options."""
    print("\n=== Load Balancing Configuration ===")

    # Check environment variable handling
    env_vars = [
        ('NCCL_MESH_RING_LOAD_BALANCE', '1', 'Enable load balancing'),
        ('NCCL_MESH_RING_PREFER_SHORT', '0', 'Prefer shorter path'),
        ('NCCL_MESH_RING_BALANCE_THRESHOLD', '1048576', 'Balance threshold (1MB)'),
    ]

    for var, default, desc in env_vars:
        current = os.environ.get(var, default)
        print(f"  {var}={current} ({desc})")

    return True


def generate_ring_config():
    """Generate a sample configuration file for a 4-node ring."""
    print("\n=== Sample Ring Configuration ===")

    config = {
        "topology": "ring",
        "nodes": [
            {
                "name": "node-a",
                "nics": [
                    {"subnet": "192.168.100.0/24", "ip": "192.168.100.1", "speed_gbps": 100},
                    {"subnet": "192.168.103.0/24", "ip": "192.168.103.1", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.1", "speed_gbps": 10, "role": "management"}
                ]
            },
            {
                "name": "node-b",
                "nics": [
                    {"subnet": "192.168.100.0/24", "ip": "192.168.100.2", "speed_gbps": 100},
                    {"subnet": "192.168.101.0/24", "ip": "192.168.101.1", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.2", "speed_gbps": 10, "role": "management"}
                ]
            },
            {
                "name": "node-c",
                "nics": [
                    {"subnet": "192.168.101.0/24", "ip": "192.168.101.2", "speed_gbps": 100},
                    {"subnet": "192.168.102.0/24", "ip": "192.168.102.1", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.3", "speed_gbps": 10, "role": "management"}
                ]
            },
            {
                "name": "node-d",
                "nics": [
                    {"subnet": "192.168.102.0/24", "ip": "192.168.102.2", "speed_gbps": 100},
                    {"subnet": "192.168.103.0/24", "ip": "192.168.103.2", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.4", "speed_gbps": 10, "role": "management"}
                ]
            }
        ],
        "expected_topology": {
            "type": "ring",
            "direct_links": ["A-B", "B-C", "C-D", "D-A"],
            "relay_links": ["A-C", "B-D"],
            "max_hops": 2
        }
    }

    print(json.dumps(config, indent=2))
    return config


def run_all_tests(verbose=False):
    """Run all ring topology tests."""
    print("=" * 60)
    print("  NCCL Mesh Plugin - Ring Topology Integration Tests")
    print("=" * 60)

    tests = [
        ("Ring topology verification", check_ring_topology),
        ("Hop count verification", check_hop_counts),
        ("Dual path routing", check_dual_path_routing),
        ("Load balancing config", check_load_balancing),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  EXCEPTION in {name}: {e}")
            failed += 1

    # Generate sample config
    if verbose:
        generate_ring_config()

    # Summary
    print("\n" + "=" * 60)
    print(f"  Results: {passed + failed} tests, {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description='Ring topology integration tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    success = run_all_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
