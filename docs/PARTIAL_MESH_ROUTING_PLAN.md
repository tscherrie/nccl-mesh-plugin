# Partial Mesh Routing Implementation Plan

This document outlines the plan for implementing partial mesh routing (ring and line topologies) in the NCCL Mesh Plugin.

## Executive Summary

The current NCCL Mesh Plugin supports **full mesh** topologies where every node pair has a direct RDMA connection. This limits scalability to 3 nodes with 2 NICs each. This plan describes how to extend the plugin to support **partial mesh** topologies (ring, line) where non-adjacent nodes communicate through intermediate relay nodes.

## Problem Statement

### Current Limitations

```
Full Mesh (3 nodes) - Currently Supported
        Node A
       /      \
   direct    direct
    link      link
     /          \
  Node B ------ Node C
       direct link

Each pair: Direct RDMA connection on separate subnet
Requirement: N-1 NICs per node for N nodes
```

### Target Topologies

**Ring Topology (4+ nodes)**
```
    Node A -------- Node B
      |               |
      |               |
    Node D -------- Node C

A↔B, B↔C, C↔D, D↔A: Direct links
A↔C, B↔D: Must relay through neighbors (2 hops)
```

**Line Topology (any number of nodes)**
```
Node A ---- Node B ---- Node C ---- Node D

A↔B, B↔C, C↔D: Direct links
A↔C: Relay through B (2 hops)
A↔D: Relay through B and C (3 hops)
```

## Architecture Overview

### Design Principles

1. **Minimize Changes to Existing Code**: Extend, don't rewrite the current architecture
2. **Transparent to NCCL**: The routing layer is internal to the plugin
3. **Automatic Topology Discovery**: No manual configuration required
4. **Fallback to Direct**: Always prefer direct connections when available
5. **Configurable**: Allow users to disable relay routing if not needed

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NCCL Application                          │
├─────────────────────────────────────────────────────────────┤
│                   NCCL Framework                             │
├─────────────────────────────────────────────────────────────┤
│                 NCCL Net Plugin API                          │
│  (listen, connect, accept, send, recv, test, close)         │
├─────────────────────────────────────────────────────────────┤
│                  Mesh Plugin Router (NEW)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Topology Discovery │ Routing Table │ Path Selection │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│               Mesh Plugin Core (EXISTING)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Subnet-Aware NIC │ QP Management │ Connection Pool  │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    RDMA Verbs API                            │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Topology Discovery and Routing Table

**Goal**: Discover the network topology and build a routing table at initialization.

#### 1.1 Data Structures

Add to `mesh_plugin.h`:

```c
// Maximum nodes and hops for routing
#define MESH_MAX_NODES 16
#define MESH_MAX_HOPS 8
#define MESH_INVALID_NODE 0xFF

// Topology types
enum mesh_topology_type {
    MESH_TOPO_UNKNOWN = 0,
    MESH_TOPO_FULL_MESH,    // All nodes directly connected
    MESH_TOPO_RING,         // Circular: each node has 2 neighbors
    MESH_TOPO_LINE,         // Linear: endpoints have 1 neighbor, others have 2
    MESH_TOPO_PARTIAL,      // Some direct connections, some relayed
};

// Node identity (exchanged during topology discovery)
struct mesh_node_id {
    uint32_t node_id;                           // Unique node identifier (hash of addresses)
    uint8_t  num_addrs;                         // Number of addresses
    struct mesh_addr_entry addrs[MESH_MAX_ADDRS]; // All addresses on this node
    uint64_t rank;                              // NCCL rank (set later)
};

// Routing table entry for one destination
struct mesh_route_entry {
    uint32_t dest_node_id;      // Destination node ID
    uint8_t  num_hops;          // Number of hops (1 = direct, 2+ = relayed)
    uint8_t  direct;            // 1 if direct connection possible, 0 if relay needed
    uint32_t next_hop_node_id;  // Next hop node ID (self if direct)
    uint32_t next_hop_ip;       // IP to connect to for next hop
    uint8_t  relay_path[MESH_MAX_HOPS];  // Full path for multi-hop (node indices)
    uint8_t  path_len;          // Number of nodes in relay_path
};

// Complete routing table
struct mesh_routing_table {
    uint32_t local_node_id;                     // Our node ID
    enum mesh_topology_type topology;           // Detected topology type
    int num_nodes;                              // Total nodes in cluster
    struct mesh_node_id nodes[MESH_MAX_NODES];  // All known nodes
    struct mesh_route_entry routes[MESH_MAX_NODES]; // Routes to each node
    int initialized;                            // 1 if routing table is ready
    pthread_mutex_t mutex;                      // Protects routing table updates
};

// Adjacency info for topology detection
struct mesh_adjacency {
    uint32_t node_id;
    int is_adjacent;            // 1 if directly reachable
    uint32_t shared_subnet;     // The subnet we share (0 if not adjacent)
};
```

#### 1.2 Topology Discovery Process

1. **Local Node ID Generation**: Generate a unique node ID from all local IP addresses
2. **Neighbor Discovery**: Exchange node IDs with all directly reachable peers
3. **Global Topology Assembly**: Gather all node IDs and adjacency info
4. **Routing Table Construction**: Compute shortest paths using BFS

```c
// New functions to add
int mesh_topology_init(void);                  // Initialize topology discovery
int mesh_discover_neighbors(void);             // Find directly connected nodes
int mesh_exchange_topology(void);              // Exchange topology with all nodes
int mesh_build_routing_table(void);            // Compute routes
int mesh_get_route(uint32_t dest_node_id, struct mesh_route_entry *route);

// Topology detection
enum mesh_topology_type mesh_detect_topology(void);
const char* mesh_topology_name(enum mesh_topology_type type);
```

#### 1.3 Implementation Details

**Node ID Generation** (in `mesh_topology_init`):
```c
// Generate node ID by hashing all local IPs
uint32_t mesh_generate_node_id(void) {
    uint32_t hash = 0x811c9dc5;  // FNV-1a initial value
    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        uint32_t ip = g_mesh_state.nics[i].ip_addr;
        hash ^= ip;
        hash *= 0x01000193;  // FNV-1a prime
    }
    return hash;
}
```

**Neighbor Discovery** (part of `mesh_listen`/`mesh_connect`):
- When a connection is established, exchange `mesh_node_id` structures
- Build adjacency list of directly reachable nodes
- This happens naturally during NCCL's init phase

**Routing Table Construction** (BFS for shortest paths):
```c
int mesh_build_routing_table(void) {
    // For each destination node:
    //   1. If directly adjacent, route = direct
    //   2. Else, BFS to find shortest path through neighbors
    //   3. Store next_hop and full path
}
```

#### 1.4 Configuration

New environment variables:
```
NCCL_MESH_ENABLE_RELAY=1      # Enable relay routing (default: 1)
NCCL_MESH_MAX_HOPS=4          # Maximum relay hops (default: 4)
NCCL_MESH_TOPOLOGY=auto       # auto, ring, line, full (default: auto)
```

---

### Phase 2: Relay Communication Layer

**Goal**: Enable communication between non-adjacent nodes through relay.

#### 2.1 Relay Mode Selection

Two approaches to consider:

**Option A: Store-and-Forward (Simpler, Higher Latency)**
- Relay node receives complete message, then forwards
- Pros: Simpler implementation, works with existing QP model
- Cons: 2x latency per hop, requires buffer space

**Option B: Cut-Through Forwarding (Complex, Lower Latency)**
- Relay node forwards packets as they arrive
- Pros: Lower latency, pipeline-friendly
- Cons: Requires packet-level handling, more complex

**Recommendation**: Start with **Store-and-Forward** for Phase 2, consider Cut-Through for Phase 3 optimization.

#### 2.2 Relay Data Structures

```c
// Relay session between two non-adjacent nodes
struct mesh_relay_session {
    uint32_t src_node_id;       // Original sender
    uint32_t dst_node_id;       // Final destination
    uint32_t relay_id;          // Unique session ID

    // Connections
    struct mesh_recv_comm *from_prev;  // Receive from previous hop
    struct mesh_send_comm *to_next;    // Send to next hop

    // Buffering for store-and-forward
    void *relay_buffer;         // Intermediate buffer
    size_t buffer_size;         // Buffer size
    struct ibv_mr *buffer_mr;   // Registered for RDMA

    // State
    int active;
    pthread_t relay_thread;     // Background relay thread
};

// Relay state per node (nodes that act as relays)
struct mesh_relay_state {
    struct mesh_relay_session sessions[MESH_MAX_NODES * MESH_MAX_NODES];
    int num_sessions;
    pthread_mutex_t mutex;
    int relay_enabled;
};
```

#### 2.3 Modified Connection Flow

**For Non-Adjacent Nodes**:

```
Original (Direct):
  Sender → mesh_connect() → Direct RDMA → Receiver

With Relay:
  Sender → mesh_connect() → Route lookup →
    If direct: Direct RDMA to receiver
    If relay:  Establish relay chain
      Sender → RDMA → Relay1 → RDMA → Relay2 → ... → Receiver
```

**Modified `mesh_connect`**:
```c
static ncclResult_t mesh_connect(int dev, void *handle, void **sendComm, ...) {
    // 1. Check if direct connection possible (existing code)
    struct mesh_nic *nic = find_nic_for_peer(handle);

    if (nic) {
        // Direct connection - use existing code path
        return mesh_connect_direct(dev, handle, sendComm, nic);
    }

    // 2. No direct connection - need relay
    if (!g_mesh_state.relay_enabled) {
        MESH_WARN("No direct path and relay disabled");
        return ncclSystemError;
    }

    // 3. Look up route in routing table
    struct mesh_route_entry route;
    if (mesh_get_route(peer_node_id, &route) != 0) {
        MESH_WARN("No route to peer");
        return ncclSystemError;
    }

    // 4. Establish relay connection
    return mesh_connect_relay(dev, handle, sendComm, &route);
}
```

#### 2.4 Relay Send/Receive

**Relayed Send** (`mesh_isend` modification):
```c
// For relayed connections, send to first hop with routing header
struct mesh_relay_header {
    uint32_t magic;             // RELAY_MAGIC
    uint32_t src_node_id;       // Original sender
    uint32_t dst_node_id;       // Final destination
    uint32_t relay_id;          // Session ID
    uint16_t hop_count;         // Current hop (incremented at each relay)
    uint16_t total_hops;        // Total hops in path
    uint32_t payload_size;      // Size of actual data
};

ncclResult_t mesh_isend_relay(void *sendComm, void *data, int size, ...) {
    struct mesh_send_comm *comm = sendComm;

    // Prepend relay header
    struct mesh_relay_header hdr = {
        .magic = RELAY_MAGIC,
        .src_node_id = g_routing_table.local_node_id,
        .dst_node_id = comm->relay_dst_node_id,
        .payload_size = size,
        ...
    };

    // Send header + data to first hop
    // (Could use scatter-gather for efficiency)
}
```

**Relay Forwarding** (new background service):
```c
void* relay_service_thread(void *arg) {
    while (!stop_requested) {
        // Poll all relay sessions for incoming data
        for (int i = 0; i < num_relay_sessions; i++) {
            if (check_incoming_relay_data(session[i])) {
                // Receive data from previous hop
                mesh_recv_blocking(session->from_prev, buffer, &size);

                // Check header - is this the final destination?
                struct mesh_relay_header *hdr = buffer;
                if (hdr->dst_node_id == local_node_id) {
                    // Deliver to local application
                    deliver_to_app(hdr, buffer + sizeof(*hdr));
                } else {
                    // Forward to next hop
                    hdr->hop_count++;
                    mesh_send_blocking(session->to_next, buffer, size);
                }
            }
        }
    }
}
```

#### 2.5 Relay Session Setup Protocol

When connecting to a non-adjacent peer:

1. **Initiator** sends `RELAY_SETUP_REQ` to first-hop neighbor
2. **Relay node** checks if it can reach the destination:
   - If direct: forwards `RELAY_SETUP_REQ` to destination
   - If not: forwards to next-hop relay
3. **Destination** receives `RELAY_SETUP_REQ`, sends `RELAY_SETUP_ACK` back
4. Each relay node in the path sets up forwarding state
5. **Initiator** receives `RELAY_SETUP_ACK`, connection is ready

---

### Phase 3: Ring and Line Topology Optimizations

**Goal**: Optimize for specific topology patterns.

#### 3.1 Ring Topology Optimizations

```
Ring: A - B - C - D - A

Communication patterns:
  A↔B: 1 hop (direct)
  A↔C: 2 hops (A→B→C or A→D→C)
  A↔D: 1 hop (direct)
  B↔D: 2 hops (B→A→D or B→C→D)
```

**Optimizations**:
1. **Dual-Path Routing**: For 2-hop destinations, two paths exist (clockwise/counterclockwise)
   - Balance load across both paths
   - Failover if one path congested

2. **Ring AllReduce Optimization**: NCCL's ring AllReduce maps naturally to ring topology
   - Detect ring AllReduce pattern
   - Use direct neighbor links only (no relay needed for ring collectives!)

#### 3.2 Line Topology Optimizations

```
Line: A - B - C - D

Communication patterns:
  A↔B: 1 hop
  A↔C: 2 hops (A→B→C)
  A↔D: 3 hops (A→B→C→D)
```

**Optimizations**:
1. **Pipeline-Aware Routing**: For long chains, enable pipelining
   - Don't wait for full message before forwarding
   - Reduces effective latency for large transfers

2. **Endpoint Awareness**: Endpoints (A, D) have only one neighbor
   - Simpler routing decisions
   - Can pre-compute all paths at init

#### 3.3 Automatic Topology Detection

```c
enum mesh_topology_type mesh_detect_topology(void) {
    // Count neighbor degree for each node
    int min_degree = INT_MAX, max_degree = 0;
    int degree_2_count = 0, degree_1_count = 0;

    for (int i = 0; i < num_nodes; i++) {
        int degree = count_neighbors(nodes[i]);
        min_degree = MIN(min_degree, degree);
        max_degree = MAX(max_degree, degree);
        if (degree == 2) degree_2_count++;
        if (degree == 1) degree_1_count++;
    }

    // Classification
    if (min_degree == max_degree && min_degree == num_nodes - 1) {
        return MESH_TOPO_FULL_MESH;  // Everyone connected to everyone
    }

    if (min_degree == 2 && max_degree == 2 && num_nodes >= 3) {
        // All nodes have exactly 2 neighbors - could be ring
        if (is_cyclic()) return MESH_TOPO_RING;
    }

    if (degree_1_count == 2 && degree_2_count == num_nodes - 2) {
        // Exactly 2 endpoints, rest have 2 neighbors - line
        return MESH_TOPO_LINE;
    }

    return MESH_TOPO_PARTIAL;  // Mixed topology
}
```

---

### Phase 4: Integration and Testing

#### 4.1 Integration Points

**Modifications to Existing Functions**:

| Function | Change |
|----------|--------|
| `mesh_init` | Add topology discovery, build routing table |
| `mesh_listen` | Exchange node IDs during handshake |
| `mesh_connect` | Check routing table, use relay if needed |
| `mesh_accept` | Handle relay setup requests |
| `mesh_isend` | Add relay header for relayed connections |
| `mesh_irecv` | Strip relay header for relayed connections |
| `mesh_close_send/recv` | Tear down relay sessions |

**New Functions**:

| Function | Purpose |
|----------|---------|
| `mesh_topology_init` | Initialize topology discovery |
| `mesh_build_routing_table` | Compute routes |
| `mesh_get_route` | Look up route to destination |
| `mesh_connect_relay` | Establish relayed connection |
| `mesh_relay_service_start/stop` | Relay service management |
| `mesh_detect_topology` | Automatic topology detection |

#### 4.2 Testing Strategy

**Unit Tests**:
1. Routing table construction with various topologies
2. Path computation (BFS correctness)
3. Relay header parsing/construction
4. Node ID generation uniqueness

**Integration Tests**:
1. 4-node ring topology with `test_allreduce.py`
2. Line topology with varying lengths
3. Mixed direct/relay communication
4. Relay node failure handling

**Performance Tests**:
1. Bandwidth through relay vs. direct
2. Latency comparison (1-hop, 2-hop, 3-hop)
3. Scalability with increasing relay hops

#### 4.3 Test Configurations

**4-Node Ring Test Setup**:
```
Node A (NIC1: 192.168.100.x, NIC2: 192.168.101.x)
Node B (NIC1: 192.168.101.x, NIC2: 192.168.102.x)
Node C (NIC1: 192.168.102.x, NIC2: 192.168.103.x)
Node D (NIC1: 192.168.103.x, NIC2: 192.168.100.x)

Direct: A↔B, B↔C, C↔D, D↔A
Relay:  A↔C (via B or D), B↔D (via A or C)
```

**4-Node Line Test Setup**:
```
Node A (NIC1: 192.168.100.x)
Node B (NIC1: 192.168.100.x, NIC2: 192.168.101.x)
Node C (NIC1: 192.168.101.x, NIC2: 192.168.102.x)
Node D (NIC1: 192.168.102.x)

Direct: A↔B, B↔C, C↔D
Relay:  A↔C (via B), A↔D (via B,C), B↔D (via C)
```

---

## File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `src/mesh_routing.c` | Routing table and topology logic |
| `src/mesh_relay.c` | Relay service implementation |
| `include/mesh_routing.h` | Routing data structures |
| `tests/test_routing.c` | Unit tests for routing |
| `tests/test_ring_topo.py` | Integration test for ring |
| `tests/test_line_topo.py` | Integration test for line |

### Modified Files

| File | Changes |
|------|---------|
| `include/mesh_plugin.h` | Add routing structures, new config options |
| `src/mesh_plugin.c` | Integrate routing into connect/send/recv |
| `Makefile` | Add new source files |
| `README.md` | Document new topology support |
| `docs/SETUP.md` | Add ring/line setup instructions |

---

## Implementation Timeline (Suggested Order)

### Milestone 1: Foundation
- [ ] Add routing data structures to header
- [ ] Implement node ID generation
- [ ] Implement neighbor discovery during handshake
- [ ] Basic routing table construction

### Milestone 2: Relay Layer
- [ ] Implement store-and-forward relay
- [ ] Modify `mesh_connect` for relay path
- [ ] Add relay header to send/recv
- [ ] Relay service thread

### Milestone 3: Topology Support
- [ ] Automatic topology detection
- [ ] Ring topology optimizations
- [ ] Line topology optimizations
- [ ] Configuration options

### Milestone 4: Testing and Polish
- [ ] Unit tests for routing
- [ ] Integration tests for ring/line
- [ ] Performance benchmarking
- [ ] Documentation updates

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Relay adds significant latency | Start with store-and-forward, optimize to cut-through later |
| Relay node becomes bottleneck | Load balance across multiple paths (ring), monitor relay node CPU |
| Topology discovery race conditions | Use barriers/synchronization during init |
| Backward compatibility | Relay routing is opt-in, existing full-mesh setups unchanged |
| Memory overhead for relay buffers | Pre-allocate limited buffer pool, queue excess requests |

---

## Success Criteria

1. **4-node ring topology works**: All pairs can communicate (direct and relayed)
2. **Line topology scales**: 4+ nodes in a line can communicate
3. **Performance acceptable**: Relay adds < 2x latency overhead per hop
4. **Backward compatible**: Existing 3-node triangle mesh continues to work
5. **NCCL collectives succeed**: AllReduce, AllGather, etc. work on ring/line

---

## References

- Current codebase: `src/mesh_plugin.c` (3356 lines)
- NCCL plugin interface: `nccl/net_v9.h`
- Existing architecture: `docs/ARCHITECTURE.md`
- Hardware setup: `docs/SETUP.md`
