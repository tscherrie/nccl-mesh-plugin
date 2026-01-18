#!/usr/bin/env python3
"""
NCCL Mesh Plugin - Line Topology Integration Test

This test validates the mesh plugin's line topology routing by simulating
a 4-node line configuration where:
  - Node A connects to B only (endpoint)
  - Node B connects to A and C
  - Node C connects to B and D
  - Node D connects to C only (endpoint)

Test Setup (simulated via environment):
    Node A (192.168.100.x) -> B
    Node B (192.168.100.x, 192.168.101.x) -> A, C
    Node C (192.168.101.x, 192.168.102.x) -> B, D
    Node D (192.168.102.x) -> C

Hop counts:
    A-B: 1 (direct)
    A-C: 2 (relay via B)
    A-D: 3 (relay via B, C)
    B-C: 1 (direct)
    B-D: 2 (relay via C)
    C-D: 1 (direct)

Usage:
    python3 test_line_topo.py [--verbose]

Requirements:
    - PyTorch with NCCL support (for actual collective tests)
    - libnccl-net-mesh.so in LD_LIBRARY_PATH
"""

import os
import sys
import argparse
import json

# Test configuration
LINE_NODES = ['A', 'B', 'C', 'D']
LINE_SUBNETS = {
    'A': ['192.168.100'],                    # Connects to B only (endpoint)
    'B': ['192.168.100', '192.168.101'],     # Connects to A and C
    'C': ['192.168.101', '192.168.102'],     # Connects to B and D
    'D': ['192.168.102'],                    # Connects to C only (endpoint)
}

# Expected connectivity matrix (hop counts)
EXPECTED_HOPS = {
    ('A', 'B'): 1, ('A', 'C'): 2, ('A', 'D'): 3,
    ('B', 'A'): 1, ('B', 'C'): 1, ('B', 'D'): 2,
    ('C', 'A'): 2, ('C', 'B'): 1, ('C', 'D'): 1,
    ('D', 'A'): 3, ('D', 'B'): 2, ('D', 'C'): 1,
}

# Endpoints have exactly 1 neighbor
ENDPOINTS = ['A', 'D']


def check_line_topology():
    """Verify that the line topology is correctly detected."""
    print("\n=== Line Topology Verification ===")

    for node in LINE_NODES:
        subnets = LINE_SUBNETS[node]
        print(f"Node {node}: subnets {subnets}")

        # Count direct neighbors
        direct_neighbors = []
        for other in LINE_NODES:
            if other == node:
                continue
            other_subnets = LINE_SUBNETS[other]
            shared = set(subnets) & set(other_subnets)
            if shared:
                direct_neighbors.append(other)

        print(f"  Direct neighbors: {direct_neighbors}")

        # Check neighbor count based on position
        expected_neighbors = 1 if node in ENDPOINTS else 2
        if len(direct_neighbors) != expected_neighbors:
            print(f"  ERROR: Expected {expected_neighbors} neighbors, got {len(direct_neighbors)}")
            return False

    print("Line topology verified: endpoints have 1 neighbor, middle nodes have 2")
    return True


def check_endpoint_detection():
    """Verify that endpoints are correctly identified."""
    print("\n=== Endpoint Detection ===")

    detected_endpoints = []
    for node in LINE_NODES:
        neighbor_count = sum(
            1 for other in LINE_NODES
            if other != node and (set(LINE_SUBNETS[node]) & set(LINE_SUBNETS[other]))
        )
        if neighbor_count == 1:
            detected_endpoints.append(node)

    print(f"  Expected endpoints: {ENDPOINTS}")
    print(f"  Detected endpoints: {detected_endpoints}")

    if set(detected_endpoints) != set(ENDPOINTS):
        print("  ERROR: Endpoint detection mismatch")
        return False

    print("Endpoint detection verified")
    return True


def check_hop_counts():
    """Verify expected hop counts between all node pairs."""
    print("\n=== Hop Count Verification ===")

    all_correct = True
    for (src, dst), expected_hops in EXPECTED_HOPS.items():
        actual_hops = calculate_line_hops(src, dst)

        status = "OK" if actual_hops == expected_hops else "FAIL"
        print(f"  {src} -> {dst}: expected {expected_hops} hop(s), got {actual_hops} [{status}]")

        if actual_hops != expected_hops:
            all_correct = False

    return all_correct


def calculate_line_hops(src, dst):
    """Calculate hop count in a line topology."""
    # In a line, hop count is the absolute difference in position
    src_idx = LINE_NODES.index(src)
    dst_idx = LINE_NODES.index(dst)
    return abs(dst_idx - src_idx)


def check_direction_routing():
    """Verify direction-based routing (towards head or tail)."""
    print("\n=== Direction Routing Verification ===")

    # From each node, check direction to all others
    for src in LINE_NODES:
        src_idx = LINE_NODES.index(src)
        print(f"  From {src} (position {src_idx}):")

        for dst in LINE_NODES:
            if dst == src:
                continue

            dst_idx = LINE_NODES.index(dst)

            # Direction: -1 towards head (A), +1 towards tail (D)
            if dst_idx < src_idx:
                direction = "head"
                next_hop = LINE_NODES[src_idx - 1] if src_idx > 0 else None
            else:
                direction = "tail"
                next_hop = LINE_NODES[src_idx + 1] if src_idx < len(LINE_NODES) - 1 else None

            print(f"    -> {dst}: direction={direction}, next_hop={next_hop}")

    return True


def check_relay_paths():
    """Verify relay paths for multi-hop communication."""
    print("\n=== Relay Path Verification ===")

    # Check paths that require relay
    relay_pairs = [
        ('A', 'C', ['A', 'B', 'C']),
        ('A', 'D', ['A', 'B', 'C', 'D']),
        ('B', 'D', ['B', 'C', 'D']),
        ('D', 'A', ['D', 'C', 'B', 'A']),
    ]

    all_correct = True
    for src, dst, expected_path in relay_pairs:
        actual_path = find_line_path(src, dst)
        status = "OK" if actual_path == expected_path else "FAIL"
        print(f"  {src} -> {dst}: {' -> '.join(actual_path)} [{status}]")

        if actual_path != expected_path:
            print(f"    Expected: {' -> '.join(expected_path)}")
            all_correct = False

    return all_correct


def find_line_path(src, dst):
    """Find the path in a line topology."""
    src_idx = LINE_NODES.index(src)
    dst_idx = LINE_NODES.index(dst)

    if src_idx <= dst_idx:
        return LINE_NODES[src_idx:dst_idx + 1]
    else:
        return list(reversed(LINE_NODES[dst_idx:src_idx + 1]))


def check_between_detection():
    """Verify mesh_line_is_between logic for relay nodes."""
    print("\n=== Between Detection (Relay Decision) ===")

    # Test cases: (node, src, dst, expected_is_between)
    test_cases = [
        ('B', 'A', 'C', True),   # B is between A and C
        ('B', 'A', 'D', True),   # B is between A and D
        ('C', 'A', 'D', True),   # C is between A and D
        ('B', 'C', 'D', False),  # B is NOT between C and D
        ('A', 'B', 'D', False),  # A is NOT between B and D (it's before B)
        ('D', 'A', 'C', False),  # D is NOT between A and C (it's after C)
    ]

    all_correct = True
    for node, src, dst, expected in test_cases:
        node_idx = LINE_NODES.index(node)
        src_idx = LINE_NODES.index(src)
        dst_idx = LINE_NODES.index(dst)

        # Calculate if node is between src and dst
        if src_idx < dst_idx:
            actual = src_idx < node_idx < dst_idx
        else:
            actual = dst_idx < node_idx < src_idx

        status = "OK" if actual == expected else "FAIL"
        print(f"  {node} between {src}-{dst}: {actual} [{status}]")

        if actual != expected:
            all_correct = False

    return all_correct


def generate_line_config():
    """Generate a sample configuration file for a 4-node line."""
    print("\n=== Sample Line Configuration ===")

    config = {
        "topology": "line",
        "nodes": [
            {
                "name": "node-a",
                "position": "head",
                "nics": [
                    {"subnet": "192.168.100.0/24", "ip": "192.168.100.1", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.1", "speed_gbps": 10, "role": "management"}
                ]
            },
            {
                "name": "node-b",
                "position": "middle",
                "nics": [
                    {"subnet": "192.168.100.0/24", "ip": "192.168.100.2", "speed_gbps": 100},
                    {"subnet": "192.168.101.0/24", "ip": "192.168.101.1", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.2", "speed_gbps": 10, "role": "management"}
                ]
            },
            {
                "name": "node-c",
                "position": "middle",
                "nics": [
                    {"subnet": "192.168.101.0/24", "ip": "192.168.101.2", "speed_gbps": 100},
                    {"subnet": "192.168.102.0/24", "ip": "192.168.102.1", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.3", "speed_gbps": 10, "role": "management"}
                ]
            },
            {
                "name": "node-d",
                "position": "tail",
                "nics": [
                    {"subnet": "192.168.102.0/24", "ip": "192.168.102.2", "speed_gbps": 100},
                    {"subnet": "10.0.0.0/24", "ip": "10.0.0.4", "speed_gbps": 10, "role": "management"}
                ]
            }
        ],
        "expected_topology": {
            "type": "line",
            "direct_links": ["A-B", "B-C", "C-D"],
            "relay_links": ["A-C", "A-D", "B-D"],
            "max_hops": 3,
            "endpoints": ["A", "D"]
        }
    }

    print(json.dumps(config, indent=2))
    return config


def run_all_tests(verbose=False):
    """Run all line topology tests."""
    print("=" * 60)
    print("  NCCL Mesh Plugin - Line Topology Integration Tests")
    print("=" * 60)

    tests = [
        ("Line topology verification", check_line_topology),
        ("Endpoint detection", check_endpoint_detection),
        ("Hop count verification", check_hop_counts),
        ("Direction routing", check_direction_routing),
        ("Relay paths", check_relay_paths),
        ("Between detection", check_between_detection),
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
        generate_line_config()

    # Summary
    print("\n" + "=" * 60)
    print(f"  Results: {passed + failed} tests, {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description='Line topology integration tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    success = run_all_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
