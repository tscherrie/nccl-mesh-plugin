/*
 * NCCL Mesh Plugin - Routing and Topology Discovery
 *
 * Implements topology discovery and routing for partial mesh networks.
 * Separates high-speed RDMA fabric (100-200Gbps) from management network (10GbE).
 *
 * Key features:
 * - NIC classification by link speed (fast lane vs management)
 * - Automatic topology detection (ring, line, full mesh)
 * - BFS-based shortest path routing for non-adjacent nodes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/stat.h>
#include <arpa/inet.h>

#include "nccl/net.h"      /* NCCL log levels */
#include "mesh_plugin.h"
#include "mesh_routing.h"

/* Global routing state */
struct mesh_routing_state g_mesh_routing = {0};

/* External reference to main plugin state */
extern struct mesh_plugin_state g_mesh_state;

/* Forward declarations for internal functions */
static int routing_bfs_shortest_path(int src_idx, int dst_idx, uint8_t *path, int *path_len);
static int count_node_fast_neighbors(int node_idx);

/*
 * =============================================================================
 * NIC Classification and Speed Detection
 * =============================================================================
 */

/*
 * Get link speed in Mbps for a network interface
 * Reads from /sys/class/net/<if_name>/speed
 *
 * Returns: Speed in Mbps, or -1 on error
 */
int mesh_get_link_speed_mbps(const char *if_name) {
    char path[256];
    char buf[32];
    int fd, n;
    int speed = -1;

    if (!if_name || !if_name[0]) {
        return -1;
    }

    snprintf(path, sizeof(path), "/sys/class/net/%s/speed", if_name);

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        /* Interface might not support speed reporting */
        MESH_DEBUG("Could not read speed for %s: %s", if_name, strerror(errno));
        return -1;
    }

    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);

    if (n > 0) {
        buf[n] = '\0';
        speed = atoi(buf);
        /* -1 or 0 means unknown/down */
        if (speed <= 0) {
            speed = -1;
        }
    }

    return speed;
}

/*
 * Classify a NIC as fast lane or management based on link speed
 */
enum mesh_nic_lane mesh_classify_nic_lane(int speed_mbps) {
    if (speed_mbps <= 0) {
        return MESH_LANE_UNKNOWN;
    }

    if (speed_mbps >= MESH_SPEED_FAST_LANE_MIN) {
        return MESH_LANE_FAST;
    }

    if (speed_mbps <= MESH_SPEED_MANAGEMENT_MAX) {
        return MESH_LANE_MANAGEMENT;
    }

    /* Middle ground (25-50 Gbps) - treat as fast if closer to fast threshold */
    if (speed_mbps > (MESH_SPEED_FAST_LANE_MIN + MESH_SPEED_MANAGEMENT_MAX) / 2) {
        return MESH_LANE_FAST;
    }

    return MESH_LANE_MANAGEMENT;
}

/*
 * Get lane classification for a NIC by index
 */
enum mesh_nic_lane mesh_get_nic_lane(int nic_idx) {
    if (nic_idx < 0 || nic_idx >= g_mesh_state.num_nics) {
        return MESH_LANE_UNKNOWN;
    }
    return (enum mesh_nic_lane)g_mesh_state.nics[nic_idx].lane;
}

/*
 * Check if NIC is fast lane
 */
int mesh_is_fast_lane_nic(int nic_idx) {
    return mesh_get_nic_lane(nic_idx) == MESH_LANE_FAST;
}

/*
 * Check if NIC is management lane
 */
int mesh_is_management_nic(int nic_idx) {
    return mesh_get_nic_lane(nic_idx) == MESH_LANE_MANAGEMENT;
}

/*
 * Get string name for lane type
 */
const char* mesh_lane_name(enum mesh_nic_lane lane) {
    switch (lane) {
        case MESH_LANE_UNKNOWN:    return "unknown";
        case MESH_LANE_MANAGEMENT: return "management";
        case MESH_LANE_FAST:       return "fast";
        default:                   return "invalid";
    }
}

/*
 * =============================================================================
 * Node ID Generation
 * =============================================================================
 */

/*
 * Generate unique node ID from all local addresses
 * Uses FNV-1a hash of all fast lane IP addresses
 */
uint32_t mesh_generate_node_id(void) {
    uint32_t hash = 0x811c9dc5;  /* FNV-1a initial value */
    int found_any = 0;

    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];

        /* Include all NICs with valid IPs in the hash */
        if (nic->ip_addr != 0) {
            hash ^= nic->ip_addr;
            hash *= 0x01000193;  /* FNV-1a prime */
            found_any = 1;
        }
    }

    /* If no IPs found, use a random-ish value based on time and pid */
    if (!found_any) {
        hash ^= (uint32_t)getpid();
        hash *= 0x01000193;
        hash ^= (uint32_t)time(NULL);
        hash *= 0x01000193;
    }

    MESH_DEBUG("Generated node ID: 0x%08x", hash);
    return hash;
}

/*
 * Build local node identity structure
 * Separates fast lane and management addresses
 */
int mesh_build_local_identity(struct mesh_node_identity *identity) {
    if (!identity) {
        return -1;
    }

    memset(identity, 0, sizeof(*identity));
    identity->magic = MESH_NODE_ID_MAGIC;
    identity->node_id = g_mesh_routing.local_node_id;

    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];

        if (nic->ip_addr == 0) {
            continue;  /* Skip NICs without IP */
        }

        struct mesh_addr_entry entry = {
            .ip = htonl(nic->ip_addr),
            .mask = htonl(nic->netmask),
            .qp_num = 0,  /* Filled in during listen */
            .nic_idx = (uint8_t)i,
            .gid_index = (uint8_t)nic->gid_index,
        };

        if (nic->lane == MESH_LANE_FAST) {
            if (identity->num_fast_addrs < MESH_MAX_ADDRS) {
                identity->fast_addrs[identity->num_fast_addrs++] = entry;
            }
        } else if (nic->lane == MESH_LANE_MANAGEMENT) {
            if (identity->num_mgmt_addrs < MESH_MAX_ADDRS) {
                identity->mgmt_addrs[identity->num_mgmt_addrs++] = entry;
            }
        } else {
            /* Unknown lane - include in management by default */
            if (identity->num_mgmt_addrs < MESH_MAX_ADDRS) {
                identity->mgmt_addrs[identity->num_mgmt_addrs++] = entry;
            }
        }
    }

    MESH_INFO("Local identity: node_id=0x%08x, fast_addrs=%d, mgmt_addrs=%d",
              identity->node_id, identity->num_fast_addrs, identity->num_mgmt_addrs);

    return 0;
}

/*
 * =============================================================================
 * Routing Initialization and Shutdown
 * =============================================================================
 */

/*
 * Initialize routing subsystem
 * Should be called after mesh_init_nics() and before connections are made
 */
int mesh_routing_init(void) {
    MESH_INFO("Initializing routing subsystem");

    memset(&g_mesh_routing, 0, sizeof(g_mesh_routing));
    pthread_mutex_init(&g_mesh_routing.mutex, NULL);

    /* Classify local NICs by speed */
    if (mesh_classify_local_nics() != 0) {
        MESH_WARN("NIC classification failed, using defaults");
    }

    /* Generate our node ID */
    g_mesh_routing.local_node_id = mesh_generate_node_id();

    /* Build our identity */
    if (mesh_build_local_identity(&g_mesh_routing.local_identity) != 0) {
        MESH_WARN("Failed to build local identity");
        return -1;
    }

    /* Register ourselves as node 0 */
    if (mesh_register_node(&g_mesh_routing.local_identity) != 0) {
        MESH_WARN("Failed to register local node");
        return -1;
    }

    g_mesh_routing.initialized = 1;

    mesh_dump_nic_classification();

    return 0;
}

/*
 * Shutdown routing subsystem
 */
void mesh_routing_destroy(void) {
    MESH_INFO("Shutting down routing subsystem");

    pthread_mutex_lock(&g_mesh_routing.mutex);
    g_mesh_routing.initialized = 0;
    g_mesh_routing.discovery_complete = 0;
    g_mesh_routing.routes_valid = 0;
    pthread_mutex_unlock(&g_mesh_routing.mutex);

    pthread_mutex_destroy(&g_mesh_routing.mutex);
}

/*
 * Classify all local NICs by link speed
 */
int mesh_classify_local_nics(void) {
    int num_fast = 0;
    int num_mgmt = 0;

    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];

        /* Get link speed */
        int speed = mesh_get_link_speed_mbps(nic->if_name);
        nic->link_speed_mbps = speed;

        /* Classify lane */
        nic->lane = (int)mesh_classify_nic_lane(speed);

        if (nic->lane == MESH_LANE_FAST) {
            num_fast++;
        } else if (nic->lane == MESH_LANE_MANAGEMENT) {
            num_mgmt++;
        }

        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(nic->ip_addr, ip_str, sizeof(ip_str));

        MESH_INFO("NIC %d (%s/%s): speed=%d Mbps, lane=%s, ip=%s",
                  i, nic->dev_name, nic->if_name,
                  speed > 0 ? speed : 0,
                  mesh_lane_name((enum mesh_nic_lane)nic->lane),
                  ip_str);
    }

    g_mesh_routing.num_fast_nics = num_fast;
    g_mesh_routing.num_mgmt_nics = num_mgmt;

    MESH_INFO("NIC classification complete: %d fast lane, %d management",
              num_fast, num_mgmt);

    return 0;
}

/*
 * =============================================================================
 * Node Registration and Adjacency
 * =============================================================================
 */

/*
 * Register a discovered node
 * Called when we receive a node identity during handshake
 */
int mesh_register_node(const struct mesh_node_identity *identity) {
    if (!identity || identity->magic != MESH_NODE_ID_MAGIC) {
        MESH_WARN("Invalid node identity");
        return -1;
    }

    pthread_mutex_lock(&g_mesh_routing.mutex);

    /* Check if already registered */
    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (g_mesh_routing.nodes[i].node_id == identity->node_id) {
            /* Already known - update identity if needed */
            memcpy(&g_mesh_routing.nodes[i], identity, sizeof(*identity));
            pthread_mutex_unlock(&g_mesh_routing.mutex);
            MESH_DEBUG("Updated known node 0x%08x", identity->node_id);
            return 0;
        }
    }

    /* Add new node */
    if (g_mesh_routing.num_nodes >= MESH_MAX_NODES) {
        pthread_mutex_unlock(&g_mesh_routing.mutex);
        MESH_WARN("Maximum nodes reached, cannot register 0x%08x", identity->node_id);
        return -1;
    }

    int idx = g_mesh_routing.num_nodes++;
    memcpy(&g_mesh_routing.nodes[idx], identity, sizeof(*identity));

    pthread_mutex_unlock(&g_mesh_routing.mutex);

    MESH_INFO("Registered node %d: id=0x%08x, fast_addrs=%d, mgmt_addrs=%d",
              idx, identity->node_id, identity->num_fast_addrs, identity->num_mgmt_addrs);

    return 0;
}

/*
 * Check if a node is already known
 */
int mesh_is_node_known(uint32_t node_id) {
    pthread_mutex_lock(&g_mesh_routing.mutex);

    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (g_mesh_routing.nodes[i].node_id == node_id) {
            pthread_mutex_unlock(&g_mesh_routing.mutex);
            return 1;
        }
    }

    pthread_mutex_unlock(&g_mesh_routing.mutex);
    return 0;
}

/*
 * Get node index by ID
 * Returns -1 if not found
 */
int mesh_get_node_index(uint32_t node_id) {
    pthread_mutex_lock(&g_mesh_routing.mutex);

    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (g_mesh_routing.nodes[i].node_id == node_id) {
            pthread_mutex_unlock(&g_mesh_routing.mutex);
            return i;
        }
    }

    pthread_mutex_unlock(&g_mesh_routing.mutex);
    return -1;
}

/*
 * Record that we discovered a directly adjacent node via fast lane
 */
int mesh_record_adjacency(uint32_t peer_node_id,
                          const struct mesh_node_identity *peer_identity,
                          int is_fast_lane,
                          uint32_t shared_subnet,
                          uint8_t local_nic_idx,
                          uint8_t remote_nic_idx) {
    pthread_mutex_lock(&g_mesh_routing.mutex);

    /* Find or create adjacency entry */
    struct mesh_adjacency *adj = NULL;
    for (int i = 0; i < g_mesh_routing.num_adjacencies; i++) {
        if (g_mesh_routing.adjacencies[i].node_id == peer_node_id) {
            adj = &g_mesh_routing.adjacencies[i];
            break;
        }
    }

    if (!adj) {
        if (g_mesh_routing.num_adjacencies >= MESH_MAX_NODES) {
            pthread_mutex_unlock(&g_mesh_routing.mutex);
            MESH_WARN("Maximum adjacencies reached");
            return -1;
        }
        adj = &g_mesh_routing.adjacencies[g_mesh_routing.num_adjacencies++];
        memset(adj, 0, sizeof(*adj));
        adj->node_id = peer_node_id;
    }

    if (is_fast_lane) {
        adj->is_fast_adjacent = 1;
        adj->shared_fast_subnet = shared_subnet;
        adj->local_fast_nic_idx = local_nic_idx;
        adj->remote_fast_nic_idx = remote_nic_idx;
        MESH_INFO("Recorded fast lane adjacency: node 0x%08x via NIC %d (subnet 0x%08x)",
                  peer_node_id, local_nic_idx, shared_subnet);
    } else {
        adj->is_mgmt_adjacent = 1;
        MESH_DEBUG("Recorded management adjacency: node 0x%08x", peer_node_id);
    }

    /* Also register the node if we have its identity */
    if (peer_identity) {
        mesh_register_node(peer_identity);
    }

    pthread_mutex_unlock(&g_mesh_routing.mutex);
    return 0;
}

/*
 * =============================================================================
 * Routing Table Computation
 * =============================================================================
 */

/*
 * Build routing table using BFS for shortest paths
 * Should be called after all nodes have been discovered
 */
int mesh_build_routing_table(void) {
    pthread_mutex_lock(&g_mesh_routing.mutex);

    MESH_INFO("Building routing table for %d nodes", g_mesh_routing.num_nodes);

    /* Find our index */
    int local_idx = -1;
    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (g_mesh_routing.nodes[i].node_id == g_mesh_routing.local_node_id) {
            local_idx = i;
            break;
        }
    }

    if (local_idx < 0) {
        pthread_mutex_unlock(&g_mesh_routing.mutex);
        MESH_WARN("Local node not found in node list");
        return -1;
    }

    /* Clear existing routes */
    memset(g_mesh_routing.routes, 0, sizeof(g_mesh_routing.routes));

    /* Build route to each node */
    for (int dest_idx = 0; dest_idx < g_mesh_routing.num_nodes; dest_idx++) {
        struct mesh_route_entry *route = &g_mesh_routing.routes[dest_idx];
        struct mesh_node_identity *dest_node = &g_mesh_routing.nodes[dest_idx];

        route->dest_node_id = dest_node->node_id;
        route->dest_node_idx = (uint8_t)dest_idx;

        if (dest_idx == local_idx) {
            /* Route to self - trivial */
            route->reachable = 1;
            route->num_hops = 0;
            route->is_direct = 1;
            continue;
        }

        /* Check if directly adjacent via fast lane */
        int is_adjacent = 0;
        struct mesh_adjacency *adj = NULL;
        for (int i = 0; i < g_mesh_routing.num_adjacencies; i++) {
            if (g_mesh_routing.adjacencies[i].node_id == dest_node->node_id &&
                g_mesh_routing.adjacencies[i].is_fast_adjacent) {
                is_adjacent = 1;
                adj = &g_mesh_routing.adjacencies[i];
                break;
            }
        }

        if (is_adjacent && adj) {
            /* Direct route via fast lane */
            route->reachable = 1;
            route->num_hops = 1;
            route->is_direct = 1;
            route->local_nic_idx = adj->local_fast_nic_idx;
            route->remote_nic_idx = adj->remote_fast_nic_idx;

            /* Find the IP address to connect to */
            for (int a = 0; a < dest_node->num_fast_addrs; a++) {
                uint32_t peer_ip = ntohl(dest_node->fast_addrs[a].ip);
                uint32_t peer_mask = ntohl(dest_node->fast_addrs[a].mask);
                uint32_t peer_subnet = peer_ip & peer_mask;
                if (peer_subnet == adj->shared_fast_subnet) {
                    route->direct_ip = peer_ip;
                    break;
                }
            }

            MESH_DEBUG("Route to node %d (0x%08x): direct via NIC %d",
                       dest_idx, dest_node->node_id, route->local_nic_idx);
        } else {
            /* Need relay - find shortest path */
            uint8_t path[MESH_MAX_HOPS];
            int path_len = 0;

            if (routing_bfs_shortest_path(local_idx, dest_idx, path, &path_len) == 0 && path_len > 0) {
                route->reachable = 1;
                route->num_hops = (uint8_t)path_len;
                route->is_direct = 0;
                route->path_len = (uint8_t)path_len;

                /* Copy path */
                for (int p = 0; p < path_len && p < MESH_MAX_HOPS; p++) {
                    route->relay_path[p] = path[p];
                }

                /* Next hop is first node in path after us */
                if (path_len >= 2) {
                    int next_hop_idx = path[1];
                    struct mesh_node_identity *next_node = &g_mesh_routing.nodes[next_hop_idx];
                    route->next_hop_node_id = next_node->node_id;

                    /* Find adjacency to next hop */
                    for (int i = 0; i < g_mesh_routing.num_adjacencies; i++) {
                        if (g_mesh_routing.adjacencies[i].node_id == next_node->node_id &&
                            g_mesh_routing.adjacencies[i].is_fast_adjacent) {
                            struct mesh_adjacency *next_adj = &g_mesh_routing.adjacencies[i];
                            route->next_hop_nic_idx = next_adj->local_fast_nic_idx;

                            /* Find IP for next hop */
                            uint32_t shared = next_adj->shared_fast_subnet;
                            for (int a = 0; a < next_node->num_fast_addrs; a++) {
                                uint32_t peer_ip = ntohl(next_node->fast_addrs[a].ip);
                                uint32_t peer_mask = ntohl(next_node->fast_addrs[a].mask);
                                if ((peer_ip & peer_mask) == shared) {
                                    route->next_hop_ip = peer_ip;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }

                MESH_DEBUG("Route to node %d (0x%08x): relay, %d hops, next_hop=0x%08x",
                           dest_idx, dest_node->node_id, path_len, route->next_hop_node_id);
            } else {
                /* No route found */
                route->reachable = 0;
                MESH_WARN("No route to node %d (0x%08x)", dest_idx, dest_node->node_id);
            }
        }
    }

    g_mesh_routing.routes_valid = 1;

    /* Detect topology */
    g_mesh_routing.fast_topology = mesh_detect_topology();

    pthread_mutex_unlock(&g_mesh_routing.mutex);

    MESH_INFO("Routing table built: topology=%s", mesh_topology_name(g_mesh_routing.fast_topology));
    mesh_dump_routing_table();

    /* Initialize topology-specific optimizations (ring dual-path, line endpoint detection) */
    if (mesh_topo_optimize_init() != 0) {
        MESH_WARN("Topology optimization init failed, using basic routing");
        /* Non-fatal - basic BFS routing will still work */
    }

    return 0;
}

/*
 * BFS shortest path algorithm
 * Returns path from src to dst (including both endpoints)
 */
static int routing_bfs_shortest_path(int src_idx, int dst_idx, uint8_t *path, int *path_len) {
    if (src_idx == dst_idx) {
        path[0] = (uint8_t)src_idx;
        *path_len = 1;
        return 0;
    }

    int num_nodes = g_mesh_routing.num_nodes;
    if (num_nodes <= 0 || src_idx < 0 || dst_idx < 0 ||
        src_idx >= num_nodes || dst_idx >= num_nodes) {
        return -1;
    }

    /* BFS state */
    int visited[MESH_MAX_NODES] = {0};
    int parent[MESH_MAX_NODES];
    int queue[MESH_MAX_NODES];
    int head = 0, tail = 0;

    for (int i = 0; i < MESH_MAX_NODES; i++) {
        parent[i] = -1;
    }

    /* Start BFS from source */
    visited[src_idx] = 1;
    queue[tail++] = src_idx;

    while (head < tail) {
        int curr = queue[head++];

        if (curr == dst_idx) {
            /* Found destination - reconstruct path */
            int temp_path[MESH_MAX_HOPS];
            int len = 0;
            int node = dst_idx;

            while (node != -1 && len < MESH_MAX_HOPS) {
                temp_path[len++] = node;
                node = parent[node];
            }

            /* Reverse path */
            *path_len = len;
            for (int i = 0; i < len; i++) {
                path[i] = (uint8_t)temp_path[len - 1 - i];
            }

            return 0;
        }

        /* Explore neighbors (fast lane adjacencies only) */
        uint32_t curr_node_id = g_mesh_routing.nodes[curr].node_id;

        for (int i = 0; i < g_mesh_routing.num_adjacencies; i++) {
            struct mesh_adjacency *adj = &g_mesh_routing.adjacencies[i];

            /* Check if this adjacency involves current node */
            /* Note: We need to check both directions */

            /* If current node is our local node, check direct adjacencies */
            if (curr_node_id == g_mesh_routing.local_node_id) {
                /* Our adjacencies are recorded from our perspective */
                if (adj->is_fast_adjacent) {
                    int neighbor_idx = mesh_get_node_index(adj->node_id);
                    if (neighbor_idx >= 0 && !visited[neighbor_idx]) {
                        visited[neighbor_idx] = 1;
                        parent[neighbor_idx] = curr;
                        queue[tail++] = neighbor_idx;
                    }
                }
            }
        }

        /* Also check if other nodes are adjacent to current node
         * This requires checking each node's adjacency to curr */
        for (int n = 0; n < num_nodes; n++) {
            if (visited[n]) continue;

            /* Check if node n is adjacent to curr via fast lane */
            /* For now, we use a simplified model where adjacency is symmetric */
            /* In a real implementation, we'd need bidirectional adjacency info */

            /* Check if curr has an adjacency to n */
            struct mesh_node_identity *curr_node = &g_mesh_routing.nodes[curr];
            struct mesh_node_identity *other_node = &g_mesh_routing.nodes[n];

            /* Check if they share a fast lane subnet */
            for (int a = 0; a < curr_node->num_fast_addrs; a++) {
                uint32_t curr_ip = ntohl(curr_node->fast_addrs[a].ip);
                uint32_t curr_mask = ntohl(curr_node->fast_addrs[a].mask);
                uint32_t curr_subnet = curr_ip & curr_mask;

                for (int b = 0; b < other_node->num_fast_addrs; b++) {
                    uint32_t other_ip = ntohl(other_node->fast_addrs[b].ip);
                    uint32_t other_mask = ntohl(other_node->fast_addrs[b].mask);
                    uint32_t other_subnet = other_ip & other_mask;

                    if (curr_subnet == other_subnet && curr_mask == other_mask) {
                        /* Same subnet - they are adjacent */
                        if (!visited[n]) {
                            visited[n] = 1;
                            parent[n] = curr;
                            queue[tail++] = n;
                        }
                        goto next_node;
                    }
                }
            }
            next_node:;
        }
    }

    /* No path found */
    *path_len = 0;
    return -1;
}

/*
 * Get route to a destination node
 */
int mesh_get_route(uint32_t dest_node_id, struct mesh_route_entry *route) {
    if (!route) {
        return -1;
    }

    pthread_mutex_lock(&g_mesh_routing.mutex);

    if (!g_mesh_routing.routes_valid) {
        pthread_mutex_unlock(&g_mesh_routing.mutex);
        MESH_DEBUG("Routing table not yet valid");
        return -1;
    }

    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (g_mesh_routing.routes[i].dest_node_id == dest_node_id) {
            memcpy(route, &g_mesh_routing.routes[i], sizeof(*route));
            pthread_mutex_unlock(&g_mesh_routing.mutex);
            return route->reachable ? 0 : -1;
        }
    }

    pthread_mutex_unlock(&g_mesh_routing.mutex);
    return -1;
}

/*
 * Check if we have a direct fast lane connection to a node
 */
int mesh_has_direct_route(uint32_t dest_node_id) {
    struct mesh_route_entry route;
    if (mesh_get_route(dest_node_id, &route) == 0) {
        return route.is_direct;
    }
    return 0;
}

/*
 * Check if we need relay to reach a node
 */
int mesh_needs_relay(uint32_t dest_node_id) {
    struct mesh_route_entry route;
    if (mesh_get_route(dest_node_id, &route) == 0) {
        return !route.is_direct;
    }
    return 1;  /* Unknown nodes need relay by default */
}

/*
 * =============================================================================
 * Topology Detection
 * =============================================================================
 */

/*
 * Count fast lane neighbors for a node
 */
static int count_node_fast_neighbors(int node_idx) {
    if (node_idx < 0 || node_idx >= g_mesh_routing.num_nodes) {
        return 0;
    }

    struct mesh_node_identity *node = &g_mesh_routing.nodes[node_idx];
    int count = 0;

    /* Count nodes that share a fast lane subnet with this node */
    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (i == node_idx) continue;

        struct mesh_node_identity *other = &g_mesh_routing.nodes[i];

        /* Check for shared fast lane subnet */
        for (int a = 0; a < node->num_fast_addrs; a++) {
            uint32_t node_ip = ntohl(node->fast_addrs[a].ip);
            uint32_t node_mask = ntohl(node->fast_addrs[a].mask);
            uint32_t node_subnet = node_ip & node_mask;

            for (int b = 0; b < other->num_fast_addrs; b++) {
                uint32_t other_ip = ntohl(other->fast_addrs[b].ip);
                uint32_t other_mask = ntohl(other->fast_addrs[b].mask);
                uint32_t other_subnet = other_ip & other_mask;

                if (node_subnet == other_subnet && node_mask == other_mask) {
                    count++;
                    goto found_neighbor;
                }
            }
        }
        found_neighbor:;
    }

    return count;
}

/*
 * Detect the fast lane topology type
 */
enum mesh_topology_type mesh_detect_topology(void) {
    int num_nodes = g_mesh_routing.num_nodes;

    if (num_nodes <= 1) {
        return MESH_TOPO_UNKNOWN;
    }

    /* Count neighbor degrees */
    int degrees[MESH_MAX_NODES];
    int min_degree = num_nodes;
    int max_degree = 0;
    int degree_1_count = 0;
    int degree_2_count = 0;

    for (int i = 0; i < num_nodes; i++) {
        degrees[i] = count_node_fast_neighbors(i);
        if (degrees[i] < min_degree) min_degree = degrees[i];
        if (degrees[i] > max_degree) max_degree = degrees[i];
        if (degrees[i] == 1) degree_1_count++;
        if (degrees[i] == 2) degree_2_count++;

        MESH_DEBUG("Node %d (0x%08x): degree=%d",
                   i, g_mesh_routing.nodes[i].node_id, degrees[i]);
    }

    MESH_INFO("Topology analysis: nodes=%d, min_degree=%d, max_degree=%d",
              num_nodes, min_degree, max_degree);

    /* Classification */
    if (min_degree == max_degree && min_degree == num_nodes - 1) {
        /* Everyone connected to everyone else */
        MESH_INFO("Detected topology: FULL MESH");
        return MESH_TOPO_FULL_MESH;
    }

    if (num_nodes >= 3 && min_degree == 2 && max_degree == 2) {
        /* All nodes have exactly 2 neighbors - ring */
        MESH_INFO("Detected topology: RING");
        return MESH_TOPO_RING;
    }

    if (num_nodes >= 2 && degree_1_count == 2 && degree_2_count == num_nodes - 2) {
        /* Exactly 2 endpoints (degree 1), rest have degree 2 - line */
        MESH_INFO("Detected topology: LINE");
        return MESH_TOPO_LINE;
    }

    if (num_nodes >= 2 && degree_1_count == num_nodes - 1 && max_degree == num_nodes - 1) {
        /* One node connected to all others, others only to that node - star */
        MESH_INFO("Detected topology: STAR");
        return MESH_TOPO_STAR;
    }

    /* Mixed/partial topology */
    MESH_INFO("Detected topology: PARTIAL");
    return MESH_TOPO_PARTIAL;
}

/*
 * Get string name for topology type
 */
const char* mesh_topology_name(enum mesh_topology_type topo) {
    switch (topo) {
        case MESH_TOPO_UNKNOWN:    return "unknown";
        case MESH_TOPO_FULL_MESH:  return "full_mesh";
        case MESH_TOPO_RING:       return "ring";
        case MESH_TOPO_LINE:       return "line";
        case MESH_TOPO_STAR:       return "star";
        case MESH_TOPO_PARTIAL:    return "partial";
        default:                   return "invalid";
    }
}

/*
 * Check if topology is a ring
 */
int mesh_is_ring_topology(void) {
    return g_mesh_routing.fast_topology == MESH_TOPO_RING;
}

/*
 * Check if topology is a line
 */
int mesh_is_line_topology(void) {
    return g_mesh_routing.fast_topology == MESH_TOPO_LINE;
}

/*
 * =============================================================================
 * Enhanced NIC Lookup (Lane-Aware)
 * =============================================================================
 */

/*
 * Find fast lane NIC that can reach peer IP
 */
struct mesh_nic* mesh_find_fast_nic_for_ip(uint32_t peer_ip) {
    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];

        /* Only consider fast lane NICs */
        if (nic->lane != MESH_LANE_FAST) {
            continue;
        }

        uint32_t peer_subnet = peer_ip & nic->netmask;
        if (peer_subnet == nic->subnet) {
            MESH_DEBUG("Found fast lane NIC %s for peer IP 0x%x",
                       nic->dev_name, peer_ip);
            return nic;
        }
    }

    return NULL;
}

/*
 * Find management NIC that can reach peer IP
 */
struct mesh_nic* mesh_find_mgmt_nic_for_ip(uint32_t peer_ip) {
    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];

        /* Only consider management NICs */
        if (nic->lane != MESH_LANE_MANAGEMENT) {
            continue;
        }

        uint32_t peer_subnet = peer_ip & nic->netmask;
        if (peer_subnet == nic->subnet) {
            MESH_DEBUG("Found management NIC %s for peer IP 0x%x",
                       nic->dev_name, peer_ip);
            return nic;
        }
    }

    return NULL;
}

/*
 * Find any NIC that can reach peer IP (prefers fast lane)
 */
struct mesh_nic* mesh_find_any_nic_for_ip(uint32_t peer_ip) {
    /* Try fast lane first */
    struct mesh_nic *nic = mesh_find_fast_nic_for_ip(peer_ip);
    if (nic) {
        return nic;
    }

    /* Fall back to management */
    nic = mesh_find_mgmt_nic_for_ip(peer_ip);
    if (nic) {
        MESH_DEBUG("Using management NIC for peer IP 0x%x (no fast lane available)", peer_ip);
        return nic;
    }

    /* Fall back to original function for unknown NICs */
    return mesh_find_nic_for_ip(peer_ip);
}

/*
 * =============================================================================
 * Debug and Diagnostics
 * =============================================================================
 */

/*
 * Print NIC classification to log
 */
void mesh_dump_nic_classification(void) {
    MESH_INFO("=== NIC Classification ===");
    for (int i = 0; i < g_mesh_state.num_nics; i++) {
        struct mesh_nic *nic = &g_mesh_state.nics[i];
        char ip_str[INET_ADDRSTRLEN];
        mesh_uint_to_ip(nic->ip_addr, ip_str, sizeof(ip_str));

        MESH_INFO("  NIC %d: %s (%s)", i, nic->dev_name, nic->if_name);
        MESH_INFO("    IP: %s, Speed: %d Mbps, Lane: %s",
                  ip_str,
                  nic->link_speed_mbps > 0 ? nic->link_speed_mbps : 0,
                  mesh_lane_name((enum mesh_nic_lane)nic->lane));
    }
    MESH_INFO("  Summary: %d fast lane, %d management NICs",
              g_mesh_routing.num_fast_nics, g_mesh_routing.num_mgmt_nics);
}

/*
 * Print routing table to log
 */
void mesh_dump_routing_table(void) {
    pthread_mutex_lock(&g_mesh_routing.mutex);

    MESH_INFO("=== Routing Table (local node 0x%08x) ===", g_mesh_routing.local_node_id);

    if (!g_mesh_routing.routes_valid) {
        MESH_INFO("  (routing table not yet computed)");
        pthread_mutex_unlock(&g_mesh_routing.mutex);
        return;
    }

    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        struct mesh_route_entry *route = &g_mesh_routing.routes[i];
        char dest_ip_str[INET_ADDRSTRLEN] = "N/A";

        if (route->is_direct && route->direct_ip) {
            mesh_uint_to_ip(route->direct_ip, dest_ip_str, sizeof(dest_ip_str));
        } else if (!route->is_direct && route->next_hop_ip) {
            mesh_uint_to_ip(route->next_hop_ip, dest_ip_str, sizeof(dest_ip_str));
        }

        if (route->dest_node_id == g_mesh_routing.local_node_id) {
            MESH_INFO("  Node %d: 0x%08x (self)", i, route->dest_node_id);
        } else if (route->reachable) {
            if (route->is_direct) {
                MESH_INFO("  Node %d: 0x%08x -> DIRECT via NIC %d (%s)",
                          i, route->dest_node_id, route->local_nic_idx, dest_ip_str);
            } else {
                MESH_INFO("  Node %d: 0x%08x -> RELAY %d hops, next=0x%08x (%s)",
                          i, route->dest_node_id, route->num_hops,
                          route->next_hop_node_id, dest_ip_str);
            }
        } else {
            MESH_INFO("  Node %d: 0x%08x -> UNREACHABLE", i, route->dest_node_id);
        }
    }

    pthread_mutex_unlock(&g_mesh_routing.mutex);
}

/*
 * Print topology summary to log
 */
void mesh_dump_topology(void) {
    pthread_mutex_lock(&g_mesh_routing.mutex);

    MESH_INFO("=== Topology Summary ===");
    MESH_INFO("  Local node ID: 0x%08x", g_mesh_routing.local_node_id);
    MESH_INFO("  Total nodes: %d", g_mesh_routing.num_nodes);
    MESH_INFO("  Fast lane topology: %s", mesh_topology_name(g_mesh_routing.fast_topology));
    MESH_INFO("  Fast lane NICs: %d", g_mesh_routing.num_fast_nics);
    MESH_INFO("  Management NICs: %d", g_mesh_routing.num_mgmt_nics);

    MESH_INFO("  Adjacencies (%d fast lane):", g_mesh_routing.num_adjacencies);
    for (int i = 0; i < g_mesh_routing.num_adjacencies; i++) {
        struct mesh_adjacency *adj = &g_mesh_routing.adjacencies[i];
        if (adj->is_fast_adjacent) {
            char subnet_str[INET_ADDRSTRLEN];
            mesh_uint_to_ip(adj->shared_fast_subnet, subnet_str, sizeof(subnet_str));
            MESH_INFO("    0x%08x: fast (subnet %s, local NIC %d)",
                      adj->node_id, subnet_str, adj->local_fast_nic_idx);
        }
    }

    pthread_mutex_unlock(&g_mesh_routing.mutex);
}

/*
 * =============================================================================
 * Phase 2: Relay Communication Layer Implementation
 * =============================================================================
 */

/* Global relay state */
struct mesh_relay_state g_mesh_relay = {0};

/*
 * =============================================================================
 * Relay Initialization and Shutdown
 * =============================================================================
 */

/*
 * Initialize relay subsystem
 */
int mesh_relay_init(void) {
    MESH_INFO("Initializing relay subsystem");

    memset(&g_mesh_relay, 0, sizeof(g_mesh_relay));
    pthread_mutex_init(&g_mesh_relay.sessions_mutex, NULL);
    pthread_cond_init(&g_mesh_relay.relay_cond, NULL);

    /* Read configuration */
    const char *env_val;

    /* NCCL_MESH_ENABLE_RELAY: Enable relay routing (default: 1) */
    env_val = getenv("NCCL_MESH_ENABLE_RELAY");
    g_mesh_relay.relay_enabled = env_val ? atoi(env_val) : 1;

    /* NCCL_MESH_MAX_RELAY_HOPS: Maximum relay hops (default: 4) */
    env_val = getenv("NCCL_MESH_MAX_RELAY_HOPS");
    g_mesh_relay.max_relay_hops = env_val ? atoi(env_val) : 4;
    if (g_mesh_relay.max_relay_hops > MESH_MAX_HOPS) {
        g_mesh_relay.max_relay_hops = MESH_MAX_HOPS;
    }

    /* Initialize session ID counter with node-unique seed */
    g_mesh_relay.next_session_id = (g_mesh_routing.local_node_id & 0xFFFF) << 16;

    MESH_INFO("Relay config: enabled=%d, max_hops=%d",
              g_mesh_relay.relay_enabled, g_mesh_relay.max_relay_hops);

    return 0;
}

/*
 * Shutdown relay subsystem
 */
void mesh_relay_destroy(void) {
    MESH_INFO("Shutting down relay subsystem");

    /* Stop relay service if running */
    mesh_relay_service_stop();

    /* Destroy all active sessions */
    pthread_mutex_lock(&g_mesh_relay.sessions_mutex);
    for (int i = 0; i < MESH_RELAY_MAX_SESSIONS; i++) {
        if (g_mesh_relay.sessions[i].state != MESH_RELAY_STATE_IDLE) {
            mesh_relay_session_destroy(&g_mesh_relay.sessions[i]);
        }
    }
    pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);

    pthread_mutex_destroy(&g_mesh_relay.sessions_mutex);
    pthread_cond_destroy(&g_mesh_relay.relay_cond);

    MESH_INFO("Relay shutdown complete: relayed %lu bytes, %lu messages, %lu errors",
              g_mesh_relay.total_bytes_relayed,
              g_mesh_relay.total_messages_relayed,
              g_mesh_relay.relay_errors);
}

/*
 * =============================================================================
 * Relay Service Thread
 * =============================================================================
 */

/*
 * Relay service thread function
 * Monitors active sessions and forwards data for intermediate nodes
 */
static void* relay_service_thread_func(void *arg) {
    (void)arg;

    MESH_INFO("Relay service thread started");

    while (!g_mesh_relay.thread_stop) {
        pthread_mutex_lock(&g_mesh_relay.sessions_mutex);

        /* Check all active sessions where we're an intermediate node */
        for (int i = 0; i < MESH_RELAY_MAX_SESSIONS; i++) {
            struct mesh_relay_session *session = &g_mesh_relay.sessions[i];

            if (session->state != MESH_RELAY_STATE_ACTIVE) {
                continue;
            }

            /* Only process if we're an intermediate relay node */
            if (!mesh_relay_is_intermediate(session)) {
                continue;
            }

            /* Check for incoming data to forward */
            /* In a full implementation, we'd poll the prev_hop connection here */
            /* For now, forwarding is triggered by mesh_relay_forward() calls */
        }

        pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);

        /* Sleep briefly before next check */
        usleep(1000);  /* 1ms */
    }

    MESH_INFO("Relay service thread stopped");
    return NULL;
}

/*
 * Start relay service thread
 */
int mesh_relay_service_start(void) {
    if (g_mesh_relay.thread_running) {
        return 0;  /* Already running */
    }

    g_mesh_relay.thread_stop = 0;

    if (pthread_create(&g_mesh_relay.relay_thread, NULL,
                       relay_service_thread_func, NULL) != 0) {
        MESH_WARN("Failed to create relay service thread: %s", strerror(errno));
        return -1;
    }

    g_mesh_relay.thread_running = 1;
    MESH_INFO("Relay service thread started");
    return 0;
}

/*
 * Stop relay service thread
 */
void mesh_relay_service_stop(void) {
    if (!g_mesh_relay.thread_running) {
        return;
    }

    g_mesh_relay.thread_stop = 1;
    pthread_cond_broadcast(&g_mesh_relay.relay_cond);

    pthread_join(g_mesh_relay.relay_thread, NULL);
    g_mesh_relay.thread_running = 0;

    MESH_INFO("Relay service thread stopped");
}

/*
 * =============================================================================
 * Relay Session Management
 * =============================================================================
 */

/*
 * Generate unique session ID
 */
uint32_t mesh_relay_generate_session_id(void) {
    pthread_mutex_lock(&g_mesh_relay.sessions_mutex);
    uint32_t id = g_mesh_relay.next_session_id++;
    pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
    return id;
}

/*
 * Create a new relay session (called by originator)
 */
struct mesh_relay_session* mesh_relay_session_create(
    uint32_t dst_node_id,
    const struct mesh_route_entry *route) {

    if (!route || !route->reachable || route->is_direct) {
        MESH_WARN("Invalid route for relay session");
        return NULL;
    }

    pthread_mutex_lock(&g_mesh_relay.sessions_mutex);

    /* Find free session slot */
    struct mesh_relay_session *session = NULL;
    for (int i = 0; i < MESH_RELAY_MAX_SESSIONS; i++) {
        if (g_mesh_relay.sessions[i].state == MESH_RELAY_STATE_IDLE) {
            session = &g_mesh_relay.sessions[i];
            break;
        }
    }

    if (!session) {
        pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
        MESH_WARN("No free relay session slots");
        return NULL;
    }

    /* Initialize session */
    memset(session, 0, sizeof(*session));
    session->session_id = mesh_relay_generate_session_id();
    session->src_node_id = g_mesh_routing.local_node_id;
    session->dst_node_id = dst_node_id;
    session->state = MESH_RELAY_STATE_SETUP;

    /* Copy path from route */
    session->path_len = route->path_len;
    memcpy(session->path, route->relay_path, route->path_len);
    session->my_position = 0;  /* We're the source */

    /* Set up next hop info */
    session->next_hop_node_id = route->next_hop_node_id;
    session->prev_hop_node_id = 0;  /* No previous hop for source */

    /* Allocate relay buffer */
    session->relay_buffer = malloc(MESH_RELAY_BUFFER_SIZE);
    if (!session->relay_buffer) {
        session->state = MESH_RELAY_STATE_IDLE;
        pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
        MESH_WARN("Failed to allocate relay buffer");
        return NULL;
    }
    session->buffer_size = MESH_RELAY_BUFFER_SIZE;

    /* Record timing */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    session->created_time = ts.tv_sec;
    session->last_activity = ts.tv_sec;

    g_mesh_relay.num_active_sessions++;

    pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);

    MESH_INFO("Created relay session 0x%08x: src=0x%08x -> dst=0x%08x, %d hops",
              session->session_id, session->src_node_id, session->dst_node_id,
              session->path_len);

    return session;
}

/*
 * Find existing session by ID
 */
struct mesh_relay_session* mesh_relay_session_find(uint32_t session_id) {
    pthread_mutex_lock(&g_mesh_relay.sessions_mutex);

    for (int i = 0; i < MESH_RELAY_MAX_SESSIONS; i++) {
        if (g_mesh_relay.sessions[i].state != MESH_RELAY_STATE_IDLE &&
            g_mesh_relay.sessions[i].session_id == session_id) {
            pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
            return &g_mesh_relay.sessions[i];
        }
    }

    pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
    return NULL;
}

/*
 * Find session by src/dst pair
 */
struct mesh_relay_session* mesh_relay_session_find_by_peers(
    uint32_t src_node_id,
    uint32_t dst_node_id) {

    pthread_mutex_lock(&g_mesh_relay.sessions_mutex);

    for (int i = 0; i < MESH_RELAY_MAX_SESSIONS; i++) {
        struct mesh_relay_session *s = &g_mesh_relay.sessions[i];
        if (s->state != MESH_RELAY_STATE_IDLE &&
            s->src_node_id == src_node_id &&
            s->dst_node_id == dst_node_id) {
            pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
            return s;
        }
    }

    pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
    return NULL;
}

/*
 * Destroy a relay session
 */
void mesh_relay_session_destroy(struct mesh_relay_session *session) {
    if (!session || session->state == MESH_RELAY_STATE_IDLE) {
        return;
    }

    MESH_INFO("Destroying relay session 0x%08x (relayed %lu bytes, %lu msgs)",
              session->session_id, session->bytes_relayed, session->messages_relayed);

    /* Free buffer */
    if (session->relay_buffer) {
        free(session->relay_buffer);
        session->relay_buffer = NULL;
    }

    /* Update statistics */
    g_mesh_relay.total_bytes_relayed += session->bytes_relayed;
    g_mesh_relay.total_messages_relayed += session->messages_relayed;

    /* Clear session */
    session->state = MESH_RELAY_STATE_IDLE;
    g_mesh_relay.num_active_sessions--;
}

/*
 * =============================================================================
 * Relay Helper Functions
 * =============================================================================
 */

/*
 * Check if we are the source of this relay session
 */
int mesh_relay_is_source(const struct mesh_relay_session *session) {
    if (!session) return 0;
    return session->src_node_id == g_mesh_routing.local_node_id;
}

/*
 * Check if we are the destination of this relay session
 */
int mesh_relay_is_destination(const struct mesh_relay_session *session) {
    if (!session) return 0;
    return session->dst_node_id == g_mesh_routing.local_node_id;
}

/*
 * Check if we are a relay node (intermediate)
 */
int mesh_relay_is_intermediate(const struct mesh_relay_session *session) {
    if (!session) return 0;
    return !mesh_relay_is_source(session) && !mesh_relay_is_destination(session);
}

/*
 * Get our position in the relay path
 */
int mesh_relay_get_position(const struct mesh_relay_session *session) {
    if (!session) return -1;
    return session->my_position;
}

/*
 * Get the next hop node ID for a relay path
 */
uint32_t mesh_relay_get_next_hop(const struct mesh_relay_session *session) {
    if (!session) return 0;
    return session->next_hop_node_id;
}

/*
 * Check if connection to peer requires relay
 */
int mesh_relay_needed(uint32_t peer_node_id) {
    if (!g_mesh_relay.relay_enabled) {
        return 0;
    }

    /* Check if we have a direct route */
    struct mesh_route_entry route;
    if (mesh_get_route(peer_node_id, &route) == 0) {
        return !route.is_direct;
    }

    return 1;  /* Unknown node, assume relay needed */
}

/*
 * =============================================================================
 * Relay Path Setup
 * =============================================================================
 */

/*
 * Set up relay path to destination (blocking)
 */
int mesh_relay_setup_path(uint32_t dst_node_id, struct mesh_relay_session **session_out) {
    if (!session_out) {
        return -1;
    }
    *session_out = NULL;

    if (!g_mesh_relay.relay_enabled) {
        MESH_WARN("Relay routing is disabled");
        return -1;
    }

    /* Get route to destination */
    struct mesh_route_entry route;
    if (mesh_get_route(dst_node_id, &route) != 0) {
        MESH_WARN("No route to destination 0x%08x", dst_node_id);
        return -1;
    }

    if (route.is_direct) {
        MESH_DEBUG("Direct route available to 0x%08x, no relay needed", dst_node_id);
        return -1;  /* Use direct connection instead */
    }

    if (route.num_hops > g_mesh_relay.max_relay_hops) {
        MESH_WARN("Route to 0x%08x requires %d hops (max %d)",
                  dst_node_id, route.num_hops, g_mesh_relay.max_relay_hops);
        return -1;
    }

    /* Check if session already exists */
    struct mesh_relay_session *session = mesh_relay_session_find_by_peers(
        g_mesh_routing.local_node_id, dst_node_id);

    if (session && session->state == MESH_RELAY_STATE_ACTIVE) {
        MESH_DEBUG("Reusing existing relay session 0x%08x", session->session_id);
        *session_out = session;
        return 0;
    }

    /* Create new session */
    session = mesh_relay_session_create(dst_node_id, &route);
    if (!session) {
        return -1;
    }

    /* Build relay header for setup request */
    struct mesh_relay_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = MESH_RELAY_MAGIC;
    hdr.session_id = session->session_id;
    hdr.src_node_id = session->src_node_id;
    hdr.dst_node_id = session->dst_node_id;
    hdr.msg_type = MESH_RELAY_SETUP_REQ;
    hdr.hop_count = 0;
    hdr.total_hops = session->path_len;
    memcpy(hdr.path, session->path, session->path_len);
    hdr.payload_size = 0;

    /*
     * In a full implementation, we would:
     * 1. Connect to next hop (route.next_hop_ip via route.next_hop_nic_idx)
     * 2. Send MESH_RELAY_SETUP_REQ header
     * 3. Wait for MESH_RELAY_SETUP_ACK from destination
     * 4. Mark session as ACTIVE
     *
     * For now, we mark the session as active and let the caller
     * establish the connection to the first hop.
     */
    (void)hdr;  /* Suppress unused warning for now */

    session->state = MESH_RELAY_STATE_ACTIVE;
    *session_out = session;

    MESH_INFO("Relay path setup complete: session 0x%08x, %d hops to 0x%08x",
              session->session_id, session->path_len, dst_node_id);

    return 0;
}

/*
 * Handle incoming relay setup request
 */
int mesh_relay_handle_setup_req(const struct mesh_relay_header *hdr, void *recv_comm) {
    if (!hdr || hdr->magic != MESH_RELAY_MAGIC) {
        return -1;
    }

    MESH_INFO("Received relay setup request: session=0x%08x, src=0x%08x, dst=0x%08x",
              hdr->session_id, hdr->src_node_id, hdr->dst_node_id);

    /* Find our position in the path */
    int my_pos = -1;
    for (int i = 0; i < hdr->total_hops && i < MESH_MAX_HOPS; i++) {
        int node_idx = hdr->path[i];
        if (node_idx < g_mesh_routing.num_nodes &&
            g_mesh_routing.nodes[node_idx].node_id == g_mesh_routing.local_node_id) {
            my_pos = i;
            break;
        }
    }

    if (my_pos < 0) {
        MESH_WARN("We're not in the relay path!");
        return -1;
    }

    /* Create or update session for this relay */
    pthread_mutex_lock(&g_mesh_relay.sessions_mutex);

    struct mesh_relay_session *session = NULL;
    for (int i = 0; i < MESH_RELAY_MAX_SESSIONS; i++) {
        if (g_mesh_relay.sessions[i].state == MESH_RELAY_STATE_IDLE) {
            session = &g_mesh_relay.sessions[i];
            break;
        }
    }

    if (!session) {
        pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
        MESH_WARN("No free session slots for relay");
        return -1;
    }

    /* Initialize session as intermediate/destination */
    memset(session, 0, sizeof(*session));
    session->session_id = hdr->session_id;
    session->src_node_id = hdr->src_node_id;
    session->dst_node_id = hdr->dst_node_id;
    session->state = MESH_RELAY_STATE_ACTIVE;
    session->path_len = hdr->total_hops;
    memcpy(session->path, hdr->path, hdr->total_hops);
    session->my_position = (uint8_t)my_pos;
    session->prev_hop_comm = recv_comm;

    /* Allocate buffer for forwarding */
    session->relay_buffer = malloc(MESH_RELAY_BUFFER_SIZE);
    if (!session->relay_buffer) {
        session->state = MESH_RELAY_STATE_IDLE;
        pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
        return -1;
    }
    session->buffer_size = MESH_RELAY_BUFFER_SIZE;

    /* Set up prev/next hop node IDs */
    if (my_pos > 0) {
        int prev_idx = hdr->path[my_pos - 1];
        if (prev_idx < g_mesh_routing.num_nodes) {
            session->prev_hop_node_id = g_mesh_routing.nodes[prev_idx].node_id;
        }
    }
    if (my_pos < hdr->total_hops - 1) {
        int next_idx = hdr->path[my_pos + 1];
        if (next_idx < g_mesh_routing.num_nodes) {
            session->next_hop_node_id = g_mesh_routing.nodes[next_idx].node_id;
        }
    }

    g_mesh_relay.num_active_sessions++;

    pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);

    MESH_INFO("Relay session 0x%08x: we are %s (position %d/%d)",
              session->session_id,
              mesh_relay_is_destination(session) ? "DESTINATION" : "RELAY",
              my_pos, hdr->total_hops);

    return 0;
}

/*
 * Handle incoming relay setup acknowledgment
 */
int mesh_relay_handle_setup_ack(const struct mesh_relay_header *hdr) {
    if (!hdr || hdr->magic != MESH_RELAY_MAGIC) {
        return -1;
    }

    struct mesh_relay_session *session = mesh_relay_session_find(hdr->session_id);
    if (!session) {
        MESH_WARN("Received ACK for unknown session 0x%08x", hdr->session_id);
        return -1;
    }

    session->state = MESH_RELAY_STATE_ACTIVE;
    MESH_INFO("Relay session 0x%08x activated", session->session_id);

    return 0;
}

/*
 * =============================================================================
 * Relay Data Transfer
 * =============================================================================
 */

/*
 * Relay request structure for async operations
 */
struct mesh_relay_request {
    int done;
    int error;
    size_t size;
    struct mesh_relay_session *session;
    void *underlying_request;  /* Request from underlying direct connection */
};

/*
 * Send data through relay path
 */
int mesh_relay_send(struct mesh_relay_session *session,
                    void *data, size_t size,
                    void **request) {
    if (!session || !data || !request) {
        return -1;
    }

    if (session->state != MESH_RELAY_STATE_ACTIVE) {
        MESH_WARN("Relay session 0x%08x not active", session->session_id);
        return -1;
    }

    /* Build relay header */
    struct mesh_relay_header *hdr = (struct mesh_relay_header *)session->relay_buffer;
    memset(hdr, 0, sizeof(*hdr));
    hdr->magic = MESH_RELAY_MAGIC;
    hdr->session_id = session->session_id;
    hdr->src_node_id = session->src_node_id;
    hdr->dst_node_id = session->dst_node_id;
    hdr->msg_type = MESH_RELAY_DATA;
    hdr->hop_count = 0;
    hdr->total_hops = session->path_len;
    memcpy(hdr->path, session->path, session->path_len);
    hdr->payload_size = (uint32_t)size;

    /* Copy data after header */
    size_t total_size = sizeof(*hdr) + size;
    if (total_size > session->buffer_size) {
        MESH_WARN("Relay message too large: %zu > %zu", total_size, session->buffer_size);
        return -1;
    }

    memcpy((char *)session->relay_buffer + sizeof(*hdr), data, size);

    /* Create request */
    struct mesh_relay_request *req = calloc(1, sizeof(*req));
    if (!req) {
        return -1;
    }
    req->session = session;
    req->size = size;
    req->done = 0;

    /*
     * In a full implementation, we would send to next_hop_comm here.
     * For now, we simulate completion since the actual RDMA send would
     * be handled by the underlying connection.
     */

    /* Update statistics */
    session->bytes_relayed += size;
    session->messages_relayed++;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    session->last_activity = ts.tv_sec;

    /* Mark as done for now (synchronous simulation) */
    req->done = 1;

    *request = req;

    MESH_DEBUG("Relay send: session 0x%08x, %zu bytes", session->session_id, size);
    return 0;
}

/*
 * Check if relay send completed
 */
int mesh_relay_send_test(void *request, int *done, int *size) {
    struct mesh_relay_request *req = (struct mesh_relay_request *)request;
    if (!req || !done) {
        return -1;
    }

    *done = req->done;
    if (size && req->done) {
        *size = (int)req->size;
    }

    if (req->done) {
        free(req);
    }

    return 0;
}

/*
 * Receive data from relay path
 */
int mesh_relay_recv(struct mesh_relay_session *session,
                    void *data, size_t size,
                    void **request) {
    if (!session || !data || !request) {
        return -1;
    }
    (void)data;  /* Used in full implementation */

    if (session->state != MESH_RELAY_STATE_ACTIVE) {
        MESH_WARN("Relay session 0x%08x not active", session->session_id);
        return -1;
    }

    /* Create request */
    struct mesh_relay_request *req = calloc(1, sizeof(*req));
    if (!req) {
        return -1;
    }
    req->session = session;
    req->size = size;
    req->done = 0;

    /*
     * In a full implementation, we would post a receive to prev_hop_comm here.
     * The relay service thread or a callback would complete this request
     * when data arrives.
     */

    *request = req;

    MESH_DEBUG("Relay recv posted: session 0x%08x, max %zu bytes",
               session->session_id, size);
    return 0;
}

/*
 * Check if relay receive completed
 */
int mesh_relay_recv_test(void *request, int *done, int *size) {
    struct mesh_relay_request *req = (struct mesh_relay_request *)request;
    if (!req || !done) {
        return -1;
    }

    *done = req->done;
    if (size && req->done) {
        *size = (int)req->size;
    }

    if (req->done) {
        free(req);
    }

    return 0;
}

/*
 * Forward data to next hop (called by relay service on intermediate nodes)
 */
int mesh_relay_forward(struct mesh_relay_session *session,
                       const struct mesh_relay_header *hdr,
                       void *data, size_t size) {
    if (!session || !hdr || !data) {
        return -1;
    }

    if (!mesh_relay_is_intermediate(session)) {
        MESH_WARN("Not an intermediate node for session 0x%08x", session->session_id);
        return -1;
    }

    /* Update header for next hop */
    struct mesh_relay_header fwd_hdr = *hdr;
    fwd_hdr.hop_count++;

    MESH_DEBUG("Forwarding: session 0x%08x, hop %d/%d, %zu bytes",
               session->session_id, fwd_hdr.hop_count, fwd_hdr.total_hops, size);

    /*
     * In a full implementation, we would:
     * 1. Copy header + data to relay_buffer
     * 2. Send via next_hop_comm
     */
    (void)fwd_hdr;  /* Used in full implementation */

    session->bytes_relayed += size;
    session->messages_relayed++;

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    session->last_activity = ts.tv_sec;

    return 0;
}

/*
 * =============================================================================
 * Relay Connection API (integrates with mesh_connect)
 * =============================================================================
 */

/*
 * Establish relay connection to non-adjacent peer
 */
int mesh_relay_connect(uint32_t peer_node_id, void **relay_comm) {
    if (!relay_comm) {
        return -1;
    }
    *relay_comm = NULL;

    if (!g_mesh_relay.relay_enabled) {
        MESH_WARN("Relay routing is disabled");
        return -1;
    }

    /* Set up relay path */
    struct mesh_relay_session *session = NULL;
    if (mesh_relay_setup_path(peer_node_id, &session) != 0) {
        return -1;
    }

    /* Create relay comm handle */
    struct mesh_relay_comm *comm = calloc(1, sizeof(*comm));
    if (!comm) {
        return -1;
    }

    comm->is_relay = 1;
    comm->session_id = session->session_id;
    comm->peer_node_id = peer_node_id;
    comm->session = session;
    comm->is_sender = 1;

    *relay_comm = comm;

    MESH_INFO("Relay connection established to 0x%08x (session 0x%08x)",
              peer_node_id, session->session_id);

    return 0;
}

/*
 * Accept relay connection from non-adjacent peer
 */
int mesh_relay_accept(void **relay_comm) {
    if (!relay_comm) {
        return -1;
    }
    *relay_comm = NULL;

    /*
     * In a full implementation, we would:
     * 1. Wait for MESH_RELAY_SETUP_REQ from a new source
     * 2. Create session via mesh_relay_handle_setup_req()
     * 3. Send MESH_RELAY_SETUP_ACK back
     * 4. Return relay comm handle
     *
     * For now, this is a placeholder that returns -1 since
     * relay accept is triggered by incoming setup requests
     * rather than being called directly.
     */

    MESH_DEBUG("Relay accept: waiting for incoming setup request");
    return -1;  /* No pending relay to accept */
}

/*
 * Close relay connection
 */
int mesh_relay_close(void *relay_comm) {
    struct mesh_relay_comm *comm = (struct mesh_relay_comm *)relay_comm;
    if (!comm || !comm->is_relay) {
        return -1;
    }

    MESH_INFO("Closing relay connection to 0x%08x (session 0x%08x)",
              comm->peer_node_id, comm->session_id);

    /* Destroy session if we're the owner (source) */
    if (comm->session && comm->is_sender) {
        mesh_relay_session_destroy(comm->session);
    }

    free(comm);
    return 0;
}

/*
 * =============================================================================
 * Relay Diagnostics
 * =============================================================================
 */

/*
 * Dump relay state to log
 */
void mesh_relay_dump_state(void) {
    MESH_INFO("=== Relay State ===");
    MESH_INFO("  Enabled: %d", g_mesh_relay.relay_enabled);
    MESH_INFO("  Max hops: %d", g_mesh_relay.max_relay_hops);
    MESH_INFO("  Service thread: %s", g_mesh_relay.thread_running ? "running" : "stopped");
    MESH_INFO("  Active sessions: %d", g_mesh_relay.num_active_sessions);
    MESH_INFO("  Total bytes relayed: %lu", g_mesh_relay.total_bytes_relayed);
    MESH_INFO("  Total messages relayed: %lu", g_mesh_relay.total_messages_relayed);
    MESH_INFO("  Relay errors: %lu", g_mesh_relay.relay_errors);
}

/*
 * Dump active sessions to log
 */
void mesh_relay_dump_sessions(void) {
    pthread_mutex_lock(&g_mesh_relay.sessions_mutex);

    MESH_INFO("=== Relay Sessions (%d active) ===", g_mesh_relay.num_active_sessions);

    for (int i = 0; i < MESH_RELAY_MAX_SESSIONS; i++) {
        struct mesh_relay_session *s = &g_mesh_relay.sessions[i];
        if (s->state == MESH_RELAY_STATE_IDLE) {
            continue;
        }

        const char *role = "unknown";
        if (mesh_relay_is_source(s)) role = "SOURCE";
        else if (mesh_relay_is_destination(s)) role = "DEST";
        else if (mesh_relay_is_intermediate(s)) role = "RELAY";

        const char *state_str = "unknown";
        switch (s->state) {
            case MESH_RELAY_STATE_SETUP: state_str = "setup"; break;
            case MESH_RELAY_STATE_ACTIVE: state_str = "active"; break;
            case MESH_RELAY_STATE_TEARDOWN: state_str = "teardown"; break;
            case MESH_RELAY_STATE_ERROR: state_str = "error"; break;
            default: break;
        }

        MESH_INFO("  Session 0x%08x: %s (%s)", s->session_id, role, state_str);
        MESH_INFO("    Path: 0x%08x -> 0x%08x (%d hops, pos %d)",
                  s->src_node_id, s->dst_node_id, s->path_len, s->my_position);
        MESH_INFO("    Stats: %lu bytes, %lu msgs relayed",
                  s->bytes_relayed, s->messages_relayed);
    }

    pthread_mutex_unlock(&g_mesh_relay.sessions_mutex);
}

/*
 * =============================================================================
 * Phase 3: Ring and Line Topology Optimizations
 * =============================================================================
 */

/* Global ring and line state */
struct mesh_ring_state g_mesh_ring = {0};
struct mesh_line_state g_mesh_line = {0};

/*
 * Forward declarations for internal helpers
 */
static int ring_find_node_position(uint32_t node_id);
static int line_find_node_position(uint32_t node_id);
static int find_adjacent_node(int node_idx, int exclude_idx);

/*
 * =============================================================================
 * Ring Topology Functions
 * =============================================================================
 */

/*
 * Initialize ring topology state
 */
int mesh_ring_init(void) {
    memset(&g_mesh_ring, 0, sizeof(g_mesh_ring));

    /* Check if we have a ring topology */
    if (g_mesh_routing.fast_topology != MESH_TOPO_RING) {
        MESH_DEBUG("Not a ring topology, skipping ring init");
        return 0;
    }

    g_mesh_ring.is_ring = 1;
    g_mesh_ring.ring_size = g_mesh_routing.num_nodes;

    /* Read configuration */
    const char *env_val;

    /* NCCL_MESH_RING_LOAD_BALANCE: Enable load balancing (default: 1) */
    env_val = getenv("NCCL_MESH_RING_LOAD_BALANCE");
    g_mesh_ring.load_balance_enabled = env_val ? atoi(env_val) : 1;

    /* NCCL_MESH_RING_PREFER_SHORT: Always prefer shorter path (default: 0) */
    env_val = getenv("NCCL_MESH_RING_PREFER_SHORT");
    g_mesh_ring.prefer_shorter_path = env_val ? atoi(env_val) : 0;

    /* NCCL_MESH_RING_BALANCE_THRESHOLD: Bytes diff to trigger switch (default: 1MB) */
    env_val = getenv("NCCL_MESH_RING_BALANCE_THRESHOLD");
    g_mesh_ring.balance_threshold = env_val ? (uint64_t)atol(env_val) : (1024 * 1024);

    MESH_INFO("Ring init: size=%d, load_balance=%d, prefer_short=%d, threshold=%lu",
              g_mesh_ring.ring_size, g_mesh_ring.load_balance_enabled,
              g_mesh_ring.prefer_shorter_path, g_mesh_ring.balance_threshold);

    /* Build ring order */
    if (mesh_ring_build_order() != 0) {
        MESH_WARN("Failed to build ring order");
        return -1;
    }

    /* Compute dual paths */
    if (mesh_ring_compute_dual_paths() != 0) {
        MESH_WARN("Failed to compute dual paths");
        return -1;
    }

    return 0;
}

/*
 * Build ring order from adjacency information
 * Traverses the ring starting from our node
 */
int mesh_ring_build_order(void) {
    if (!g_mesh_ring.is_ring) {
        return -1;
    }

    int num_nodes = g_mesh_routing.num_nodes;
    if (num_nodes < 3) {
        MESH_WARN("Ring requires at least 3 nodes, have %d", num_nodes);
        return -1;
    }

    /* Find our node index */
    int local_idx = mesh_get_node_index(g_mesh_routing.local_node_id);
    if (local_idx < 0) {
        MESH_WARN("Local node not found in node list");
        return -1;
    }

    /* Track visited nodes */
    int visited[MESH_MAX_NODES] = {0};
    int order_idx = 0;

    /* Start with our node */
    int curr = local_idx;
    visited[curr] = 1;
    g_mesh_ring.ring_order[order_idx++] = (uint8_t)curr;

    /* Find our two neighbors */
    int neighbor1 = find_adjacent_node(curr, -1);
    int neighbor2 = find_adjacent_node(curr, neighbor1);

    if (neighbor1 < 0 || neighbor2 < 0) {
        MESH_WARN("Could not find two neighbors for ring node %d", curr);
        return -1;
    }

    /* Record our neighbors - arbitrarily call neighbor1 CW, neighbor2 CCW */
    g_mesh_ring.our_position = 0;
    g_mesh_ring.cw_neighbor_id = g_mesh_routing.nodes[neighbor1].node_id;
    g_mesh_ring.ccw_neighbor_id = g_mesh_routing.nodes[neighbor2].node_id;

    /* Traverse clockwise (via neighbor1) */
    curr = neighbor1;
    while (!visited[curr] && order_idx < num_nodes) {
        visited[curr] = 1;
        g_mesh_ring.ring_order[order_idx++] = (uint8_t)curr;

        /* Find next unvisited neighbor */
        int next1 = find_adjacent_node(curr, -1);
        int next2 = find_adjacent_node(curr, next1);

        if (next1 >= 0 && !visited[next1]) {
            curr = next1;
        } else if (next2 >= 0 && !visited[next2]) {
            curr = next2;
        } else {
            break;  /* No more unvisited neighbors */
        }
    }

    if (order_idx != num_nodes) {
        MESH_WARN("Ring traversal incomplete: got %d nodes, expected %d", order_idx, num_nodes);
        return -1;
    }

    g_mesh_ring.ring_order_valid = 1;

    MESH_INFO("Ring order built: us at position 0, CW neighbor=0x%08x, CCW neighbor=0x%08x",
              g_mesh_ring.cw_neighbor_id, g_mesh_ring.ccw_neighbor_id);

    return 0;
}

/*
 * Find an adjacent node (fast lane), optionally excluding one
 */
static int find_adjacent_node(int node_idx, int exclude_idx) {
    if (node_idx < 0 || node_idx >= g_mesh_routing.num_nodes) {
        return -1;
    }

    struct mesh_node_identity *node = &g_mesh_routing.nodes[node_idx];

    /* Find nodes that share a fast lane subnet with this node */
    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (i == node_idx || i == exclude_idx) continue;

        struct mesh_node_identity *other = &g_mesh_routing.nodes[i];

        /* Check for shared fast lane subnet */
        for (int a = 0; a < node->num_fast_addrs; a++) {
            uint32_t node_ip = ntohl(node->fast_addrs[a].ip);
            uint32_t node_mask = ntohl(node->fast_addrs[a].mask);
            uint32_t node_subnet = node_ip & node_mask;

            for (int b = 0; b < other->num_fast_addrs; b++) {
                uint32_t other_ip = ntohl(other->fast_addrs[b].ip);
                uint32_t other_mask = ntohl(other->fast_addrs[b].mask);
                uint32_t other_subnet = other_ip & other_mask;

                if (node_subnet == other_subnet && node_mask == other_mask) {
                    return i;  /* Found adjacent node */
                }
            }
        }
    }

    return -1;  /* No adjacent node found */
}

/*
 * Find a node's position in the ring order
 */
static int ring_find_node_position(uint32_t node_id) {
    if (!g_mesh_ring.ring_order_valid) {
        return -1;
    }

    for (int i = 0; i < g_mesh_ring.ring_size; i++) {
        int node_idx = g_mesh_ring.ring_order[i];
        if (node_idx < g_mesh_routing.num_nodes &&
            g_mesh_routing.nodes[node_idx].node_id == node_id) {
            return i;
        }
    }

    return -1;
}

/*
 * Compute dual paths (CW and CCW) for all destinations in ring
 */
int mesh_ring_compute_dual_paths(void) {
    if (!g_mesh_ring.is_ring || !g_mesh_ring.ring_order_valid) {
        return -1;
    }

    int n = g_mesh_ring.ring_size;

    MESH_INFO("Computing dual paths for %d-node ring", n);

    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        struct mesh_ring_dual_path *dp = &g_mesh_ring.dual_paths[i];
        struct mesh_node_identity *dest = &g_mesh_routing.nodes[i];

        memset(dp, 0, sizeof(*dp));
        dp->dest_node_id = dest->node_id;

        if (dest->node_id == g_mesh_routing.local_node_id) {
            /* Path to self - trivial */
            dp->is_valid = 1;
            dp->cw_path_len = 0;
            dp->ccw_path_len = 0;
            continue;
        }

        int dest_pos = ring_find_node_position(dest->node_id);
        if (dest_pos < 0) {
            MESH_WARN("Destination 0x%08x not found in ring order", dest->node_id);
            continue;
        }

        /* Our position is 0 in ring_order */
        int our_pos = 0;

        /* CW distance: how many hops clockwise */
        int cw_dist = (dest_pos - our_pos + n) % n;

        /* CCW distance: how many hops counter-clockwise */
        int ccw_dist = (our_pos - dest_pos + n) % n;

        /* Build CW path */
        dp->cw_path_len = (uint8_t)cw_dist;
        for (int j = 0; j < cw_dist && j < MESH_MAX_HOPS; j++) {
            int pos = (our_pos + j + 1) % n;
            dp->cw_path[j] = g_mesh_ring.ring_order[pos];
        }

        /* CW first hop info */
        if (cw_dist > 0) {
            int cw_next_idx = g_mesh_ring.ring_order[(our_pos + 1) % n];
            dp->cw_next_hop_id = g_mesh_routing.nodes[cw_next_idx].node_id;

            /* Find IP and NIC for CW next hop */
            for (int a = 0; a < g_mesh_routing.num_adjacencies; a++) {
                if (g_mesh_routing.adjacencies[a].node_id == dp->cw_next_hop_id &&
                    g_mesh_routing.adjacencies[a].is_fast_adjacent) {
                    dp->cw_next_hop_nic = g_mesh_routing.adjacencies[a].local_fast_nic_idx;

                    /* Find peer IP on shared subnet */
                    uint32_t shared = g_mesh_routing.adjacencies[a].shared_fast_subnet;
                    struct mesh_node_identity *next = &g_mesh_routing.nodes[cw_next_idx];
                    for (int b = 0; b < next->num_fast_addrs; b++) {
                        uint32_t peer_ip = ntohl(next->fast_addrs[b].ip);
                        uint32_t peer_mask = ntohl(next->fast_addrs[b].mask);
                        if ((peer_ip & peer_mask) == shared) {
                            dp->cw_next_hop_ip = peer_ip;
                            break;
                        }
                    }
                    break;
                }
            }
        }

        /* Build CCW path */
        dp->ccw_path_len = (uint8_t)ccw_dist;
        for (int j = 0; j < ccw_dist && j < MESH_MAX_HOPS; j++) {
            int pos = (our_pos - j - 1 + n) % n;
            dp->ccw_path[j] = g_mesh_ring.ring_order[pos];
        }

        /* CCW first hop info */
        if (ccw_dist > 0) {
            int ccw_next_idx = g_mesh_ring.ring_order[(our_pos - 1 + n) % n];
            dp->ccw_next_hop_id = g_mesh_routing.nodes[ccw_next_idx].node_id;

            /* Find IP and NIC for CCW next hop */
            for (int a = 0; a < g_mesh_routing.num_adjacencies; a++) {
                if (g_mesh_routing.adjacencies[a].node_id == dp->ccw_next_hop_id &&
                    g_mesh_routing.adjacencies[a].is_fast_adjacent) {
                    dp->ccw_next_hop_nic = g_mesh_routing.adjacencies[a].local_fast_nic_idx;

                    /* Find peer IP on shared subnet */
                    uint32_t shared = g_mesh_routing.adjacencies[a].shared_fast_subnet;
                    struct mesh_node_identity *next = &g_mesh_routing.nodes[ccw_next_idx];
                    for (int b = 0; b < next->num_fast_addrs; b++) {
                        uint32_t peer_ip = ntohl(next->fast_addrs[b].ip);
                        uint32_t peer_mask = ntohl(next->fast_addrs[b].mask);
                        if ((peer_ip & peer_mask) == shared) {
                            dp->ccw_next_hop_ip = peer_ip;
                            break;
                        }
                    }
                    break;
                }
            }
        }

        /* Set initial preferred direction to shorter path */
        if (cw_dist <= ccw_dist) {
            dp->preferred = MESH_RING_DIR_CW;
        } else {
            dp->preferred = MESH_RING_DIR_CCW;
        }

        dp->is_valid = 1;

        MESH_DEBUG("Dual path to 0x%08x: CW=%d hops, CCW=%d hops, preferred=%s",
                   dest->node_id, cw_dist, ccw_dist,
                   dp->preferred == MESH_RING_DIR_CW ? "CW" : "CCW");
    }

    return 0;
}

/*
 * Get the shorter path direction to a destination
 */
enum mesh_ring_direction mesh_ring_get_shorter_direction(uint32_t dest_node_id) {
    struct mesh_ring_dual_path *dp = mesh_ring_get_dual_path(dest_node_id);
    if (!dp || !dp->is_valid) {
        return MESH_RING_DIR_NONE;
    }

    if (dp->cw_path_len <= dp->ccw_path_len) {
        return MESH_RING_DIR_CW;
    }
    return MESH_RING_DIR_CCW;
}

/*
 * Get the next hop for a given direction
 */
uint32_t mesh_ring_get_next_hop(enum mesh_ring_direction direction) {
    if (!g_mesh_ring.is_ring) {
        return 0;
    }

    switch (direction) {
        case MESH_RING_DIR_CW:
            return g_mesh_ring.cw_neighbor_id;
        case MESH_RING_DIR_CCW:
            return g_mesh_ring.ccw_neighbor_id;
        default:
            return 0;
    }
}

/*
 * Select path based on load balancing
 */
enum mesh_ring_direction mesh_ring_select_path(uint32_t dest_node_id, size_t msg_size) {
    struct mesh_ring_dual_path *dp = mesh_ring_get_dual_path(dest_node_id);
    if (!dp || !dp->is_valid) {
        return MESH_RING_DIR_NONE;
    }

    /* Trivial case: path to self */
    if (dp->cw_path_len == 0 && dp->ccw_path_len == 0) {
        return MESH_RING_DIR_NONE;
    }

    /* If one path is direct (1 hop) and other is not, prefer direct */
    if (dp->cw_path_len == 1 && dp->ccw_path_len > 1) {
        return MESH_RING_DIR_CW;
    }
    if (dp->ccw_path_len == 1 && dp->cw_path_len > 1) {
        return MESH_RING_DIR_CCW;
    }

    /* If prefer_shorter_path is set, always use shorter */
    if (g_mesh_ring.prefer_shorter_path) {
        return mesh_ring_get_shorter_direction(dest_node_id);
    }

    /* If load balancing is disabled, use preferred (shorter) path */
    if (!g_mesh_ring.load_balance_enabled) {
        return dp->preferred;
    }

    /* Load balancing: check if paths have similar length */
    if (dp->cw_path_len != dp->ccw_path_len) {
        /* Unequal paths - prefer shorter unless load is very unbalanced */
        enum mesh_ring_direction shorter = mesh_ring_get_shorter_direction(dest_node_id);
        uint64_t shorter_bytes = (shorter == MESH_RING_DIR_CW) ?
                                  dp->cw_bytes_sent : dp->ccw_bytes_sent;
        uint64_t longer_bytes = (shorter == MESH_RING_DIR_CW) ?
                                 dp->ccw_bytes_sent : dp->cw_bytes_sent;

        /* Use longer path only if shorter is very congested */
        if (shorter_bytes > longer_bytes + g_mesh_ring.balance_threshold * 2) {
            return (shorter == MESH_RING_DIR_CW) ? MESH_RING_DIR_CCW : MESH_RING_DIR_CW;
        }
        return shorter;
    }

    /* Equal length paths - balance load */
    (void)msg_size;  /* Could use message size for weighted decisions */

    if (dp->cw_bytes_sent <= dp->ccw_bytes_sent) {
        if (dp->ccw_bytes_sent - dp->cw_bytes_sent > g_mesh_ring.balance_threshold) {
            return MESH_RING_DIR_CW;
        }
    } else {
        if (dp->cw_bytes_sent - dp->ccw_bytes_sent > g_mesh_ring.balance_threshold) {
            return MESH_RING_DIR_CCW;
        }
    }

    /* Use current preferred direction */
    return dp->preferred;
}

/*
 * Get dual path entry for a destination
 */
struct mesh_ring_dual_path* mesh_ring_get_dual_path(uint32_t dest_node_id) {
    if (!g_mesh_ring.is_ring) {
        return NULL;
    }

    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        if (g_mesh_ring.dual_paths[i].dest_node_id == dest_node_id) {
            return &g_mesh_ring.dual_paths[i];
        }
    }

    return NULL;
}

/*
 * Update path statistics after sending
 */
void mesh_ring_update_stats(uint32_t dest_node_id, enum mesh_ring_direction dir, size_t bytes) {
    struct mesh_ring_dual_path *dp = mesh_ring_get_dual_path(dest_node_id);
    if (!dp || !dp->is_valid) {
        return;
    }

    switch (dir) {
        case MESH_RING_DIR_CW:
            dp->cw_bytes_sent += bytes;
            break;
        case MESH_RING_DIR_CCW:
            dp->ccw_bytes_sent += bytes;
            break;
        default:
            break;
    }

    /* Update preferred direction based on load balance */
    if (g_mesh_ring.load_balance_enabled && dp->cw_path_len == dp->ccw_path_len) {
        if (dp->cw_bytes_sent < dp->ccw_bytes_sent) {
            dp->preferred = MESH_RING_DIR_CW;
        } else {
            dp->preferred = MESH_RING_DIR_CCW;
        }
    }
}

/*
 * Get ring hop count in a direction
 */
int mesh_ring_hop_count(uint32_t dest_node_id, enum mesh_ring_direction direction) {
    struct mesh_ring_dual_path *dp = mesh_ring_get_dual_path(dest_node_id);
    if (!dp || !dp->is_valid) {
        return -1;
    }

    switch (direction) {
        case MESH_RING_DIR_CW:
            return dp->cw_path_len;
        case MESH_RING_DIR_CCW:
            return dp->ccw_path_len;
        default:
            return -1;
    }
}

/*
 * Check if we're a ring neighbor of a node
 */
int mesh_ring_is_neighbor(uint32_t node_id) {
    if (!g_mesh_ring.is_ring) {
        return 0;
    }
    return (node_id == g_mesh_ring.cw_neighbor_id ||
            node_id == g_mesh_ring.ccw_neighbor_id);
}

/*
 * Dump ring state to log
 */
void mesh_ring_dump_state(void) {
    MESH_INFO("=== Ring Topology State ===");
    MESH_INFO("  Is ring: %d", g_mesh_ring.is_ring);

    if (!g_mesh_ring.is_ring) {
        return;
    }

    MESH_INFO("  Ring size: %d", g_mesh_ring.ring_size);
    MESH_INFO("  Our position: %d", g_mesh_ring.our_position);
    MESH_INFO("  CW neighbor: 0x%08x", g_mesh_ring.cw_neighbor_id);
    MESH_INFO("  CCW neighbor: 0x%08x", g_mesh_ring.ccw_neighbor_id);
    MESH_INFO("  Load balance: %d (threshold=%lu)",
              g_mesh_ring.load_balance_enabled, g_mesh_ring.balance_threshold);
    MESH_INFO("  Prefer shorter: %d", g_mesh_ring.prefer_shorter_path);

    MESH_INFO("  Ring order (%s):", g_mesh_ring.ring_order_valid ? "valid" : "invalid");
    if (g_mesh_ring.ring_order_valid) {
        for (int i = 0; i < g_mesh_ring.ring_size; i++) {
            int idx = g_mesh_ring.ring_order[i];
            MESH_INFO("    [%d] Node %d: 0x%08x", i, idx,
                      g_mesh_routing.nodes[idx].node_id);
        }
    }

    MESH_INFO("  Dual paths:");
    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        struct mesh_ring_dual_path *dp = &g_mesh_ring.dual_paths[i];
        if (!dp->is_valid) continue;
        if (dp->dest_node_id == g_mesh_routing.local_node_id) continue;

        MESH_INFO("    -> 0x%08x: CW=%d hops (%lu B), CCW=%d hops (%lu B), pref=%s",
                  dp->dest_node_id, dp->cw_path_len, dp->cw_bytes_sent,
                  dp->ccw_path_len, dp->ccw_bytes_sent,
                  dp->preferred == MESH_RING_DIR_CW ? "CW" : "CCW");
    }
}

/*
 * =============================================================================
 * Line Topology Functions
 * =============================================================================
 */

/*
 * Find a node's position in the line order
 */
static int line_find_node_position(uint32_t node_id) {
    if (!g_mesh_line.line_order_valid) {
        return -1;
    }

    for (int i = 0; i < g_mesh_line.line_length; i++) {
        int node_idx = g_mesh_line.line_order[i];
        if (node_idx < g_mesh_routing.num_nodes &&
            g_mesh_routing.nodes[node_idx].node_id == node_id) {
            return i;
        }
    }

    return -1;
}

/*
 * Initialize line topology state
 */
int mesh_line_init(void) {
    memset(&g_mesh_line, 0, sizeof(g_mesh_line));

    /* Check if we have a line topology */
    if (g_mesh_routing.fast_topology != MESH_TOPO_LINE) {
        MESH_DEBUG("Not a line topology, skipping line init");
        return 0;
    }

    g_mesh_line.is_line = 1;
    g_mesh_line.line_length = g_mesh_routing.num_nodes;

    MESH_INFO("Line init: length=%d", g_mesh_line.line_length);

    /* Detect endpoints first */
    if (mesh_line_detect_endpoints() != 0) {
        MESH_WARN("Failed to detect line endpoints");
        return -1;
    }

    /* Build line order */
    if (mesh_line_build_order() != 0) {
        MESH_WARN("Failed to build line order");
        return -1;
    }

    return 0;
}

/*
 * Detect and record line endpoints
 * Endpoints are nodes with exactly 1 fast lane neighbor
 */
int mesh_line_detect_endpoints(void) {
    if (!g_mesh_line.is_line) {
        return -1;
    }

    int endpoints_found = 0;
    memset(&g_mesh_line.head, 0, sizeof(g_mesh_line.head));
    memset(&g_mesh_line.tail, 0, sizeof(g_mesh_line.tail));

    for (int i = 0; i < g_mesh_routing.num_nodes; i++) {
        /* Count fast lane neighbors */
        int neighbor1 = find_adjacent_node(i, -1);
        int neighbor2 = find_adjacent_node(i, neighbor1);

        if (neighbor1 >= 0 && neighbor2 < 0) {
            /* Exactly one neighbor - this is an endpoint */
            struct mesh_line_endpoint *ep;

            if (endpoints_found == 0) {
                ep = &g_mesh_line.head;
                ep->is_head = 1;
                ep->is_tail = 0;
            } else if (endpoints_found == 1) {
                ep = &g_mesh_line.tail;
                ep->is_head = 0;
                ep->is_tail = 1;
            } else {
                MESH_WARN("Line has more than 2 endpoints!");
                return -1;
            }

            ep->node_id = g_mesh_routing.nodes[i].node_id;
            ep->node_idx = i;
            ep->neighbor_id = g_mesh_routing.nodes[neighbor1].node_id;

            MESH_INFO("Found %s endpoint: node 0x%08x (idx %d), neighbor=0x%08x",
                      ep->is_head ? "head" : "tail", ep->node_id, ep->node_idx,
                      ep->neighbor_id);

            endpoints_found++;
        }
    }

    if (endpoints_found != 2) {
        MESH_WARN("Line topology requires exactly 2 endpoints, found %d", endpoints_found);
        return -1;
    }

    /* Check if we are an endpoint */
    if (g_mesh_line.head.node_id == g_mesh_routing.local_node_id ||
        g_mesh_line.tail.node_id == g_mesh_routing.local_node_id) {
        g_mesh_line.is_endpoint = 1;
        MESH_INFO("We are a line endpoint");
    }

    return 0;
}

/*
 * Build line order from head to tail
 */
int mesh_line_build_order(void) {
    if (!g_mesh_line.is_line) {
        return -1;
    }

    int num_nodes = g_mesh_routing.num_nodes;
    if (num_nodes < 2) {
        MESH_WARN("Line requires at least 2 nodes, have %d", num_nodes);
        return -1;
    }

    /* Track visited nodes */
    int visited[MESH_MAX_NODES] = {0};
    int order_idx = 0;

    /* Start from head endpoint */
    int curr = g_mesh_line.head.node_idx;
    visited[curr] = 1;
    g_mesh_line.line_order[order_idx++] = (uint8_t)curr;

    /* Traverse to tail */
    while (order_idx < num_nodes) {
        /* Find unvisited neighbor */
        int next1 = find_adjacent_node(curr, -1);
        int next2 = find_adjacent_node(curr, next1);

        int next = -1;
        if (next1 >= 0 && !visited[next1]) {
            next = next1;
        } else if (next2 >= 0 && !visited[next2]) {
            next = next2;
        }

        if (next < 0) {
            break;  /* No more unvisited neighbors */
        }

        visited[next] = 1;
        g_mesh_line.line_order[order_idx++] = (uint8_t)next;
        curr = next;
    }

    if (order_idx != num_nodes) {
        MESH_WARN("Line traversal incomplete: got %d nodes, expected %d", order_idx, num_nodes);
        return -1;
    }

    /* Verify last node is tail endpoint */
    if (g_mesh_routing.nodes[curr].node_id != g_mesh_line.tail.node_id) {
        MESH_WARN("Line traversal did not reach tail endpoint");
        /* Swap head and tail if we went the wrong way */
        struct mesh_line_endpoint temp = g_mesh_line.head;
        g_mesh_line.head = g_mesh_line.tail;
        g_mesh_line.tail = temp;
        g_mesh_line.head.is_head = 1;
        g_mesh_line.head.is_tail = 0;
        g_mesh_line.tail.is_head = 0;
        g_mesh_line.tail.is_tail = 1;
    }

    g_mesh_line.line_order_valid = 1;

    /* Find our position */
    g_mesh_line.our_position = line_find_node_position(g_mesh_routing.local_node_id);

    /* Set up our neighbors */
    if (g_mesh_line.our_position > 0) {
        int left_idx = g_mesh_line.line_order[g_mesh_line.our_position - 1];
        g_mesh_line.left_neighbor_id = g_mesh_routing.nodes[left_idx].node_id;
    }
    if (g_mesh_line.our_position < g_mesh_line.line_length - 1) {
        int right_idx = g_mesh_line.line_order[g_mesh_line.our_position + 1];
        g_mesh_line.right_neighbor_id = g_mesh_routing.nodes[right_idx].node_id;
    }

    MESH_INFO("Line order built: our position=%d, left=0x%08x, right=0x%08x",
              g_mesh_line.our_position, g_mesh_line.left_neighbor_id, g_mesh_line.right_neighbor_id);

    return 0;
}

/*
 * Check if we are a line endpoint
 */
int mesh_line_is_endpoint(void) {
    return g_mesh_line.is_line && g_mesh_line.is_endpoint;
}

/*
 * Check if a node is a line endpoint
 */
int mesh_line_node_is_endpoint(uint32_t node_id) {
    if (!g_mesh_line.is_line) {
        return 0;
    }
    return (node_id == g_mesh_line.head.node_id ||
            node_id == g_mesh_line.tail.node_id);
}

/*
 * Get direction to destination (towards head or tail)
 * Returns: -1 = towards head, 1 = towards tail, 0 = error/self
 */
int mesh_line_get_direction(uint32_t dest_node_id) {
    if (!g_mesh_line.is_line || !g_mesh_line.line_order_valid) {
        return 0;
    }

    int dest_pos = line_find_node_position(dest_node_id);
    if (dest_pos < 0) {
        return 0;
    }

    if (dest_pos < g_mesh_line.our_position) {
        return -1;  /* Towards head */
    } else if (dest_pos > g_mesh_line.our_position) {
        return 1;   /* Towards tail */
    }

    return 0;  /* Self or error */
}

/*
 * Get hop count to destination
 */
int mesh_line_hop_count(uint32_t dest_node_id) {
    if (!g_mesh_line.is_line || !g_mesh_line.line_order_valid) {
        return -1;
    }

    int dest_pos = line_find_node_position(dest_node_id);
    if (dest_pos < 0) {
        return -1;
    }

    int diff = dest_pos - g_mesh_line.our_position;
    return (diff >= 0) ? diff : -diff;
}

/*
 * Get next hop towards a destination
 */
uint32_t mesh_line_get_next_hop(uint32_t dest_node_id) {
    if (!g_mesh_line.is_line || !g_mesh_line.line_order_valid) {
        return 0;
    }

    int direction = mesh_line_get_direction(dest_node_id);

    if (direction < 0) {
        /* Towards head */
        return g_mesh_line.left_neighbor_id;
    } else if (direction > 0) {
        /* Towards tail */
        return g_mesh_line.right_neighbor_id;
    }

    return 0;  /* Self or error */
}

/*
 * Check if we're between two nodes (for relay decisions)
 * Returns 1 if we are on the path between src and dst
 */
int mesh_line_is_between(uint32_t src_node_id, uint32_t dst_node_id) {
    if (!g_mesh_line.is_line || !g_mesh_line.line_order_valid) {
        return 0;
    }

    int src_pos = line_find_node_position(src_node_id);
    int dst_pos = line_find_node_position(dst_node_id);

    if (src_pos < 0 || dst_pos < 0) {
        return 0;
    }

    int our_pos = g_mesh_line.our_position;

    /* We're between if our position is strictly between src and dst */
    if (src_pos < dst_pos) {
        return (our_pos > src_pos && our_pos < dst_pos);
    } else {
        return (our_pos > dst_pos && our_pos < src_pos);
    }
}

/*
 * Dump line state to log
 */
void mesh_line_dump_state(void) {
    MESH_INFO("=== Line Topology State ===");
    MESH_INFO("  Is line: %d", g_mesh_line.is_line);

    if (!g_mesh_line.is_line) {
        return;
    }

    MESH_INFO("  Line length: %d", g_mesh_line.line_length);
    MESH_INFO("  Head endpoint: 0x%08x (idx %d)", g_mesh_line.head.node_id, g_mesh_line.head.node_idx);
    MESH_INFO("  Tail endpoint: 0x%08x (idx %d)", g_mesh_line.tail.node_id, g_mesh_line.tail.node_idx);
    MESH_INFO("  Our position: %d (is_endpoint=%d)", g_mesh_line.our_position, g_mesh_line.is_endpoint);
    MESH_INFO("  Left neighbor: 0x%08x", g_mesh_line.left_neighbor_id);
    MESH_INFO("  Right neighbor: 0x%08x", g_mesh_line.right_neighbor_id);

    MESH_INFO("  Line order (%s):", g_mesh_line.line_order_valid ? "valid" : "invalid");
    if (g_mesh_line.line_order_valid) {
        for (int i = 0; i < g_mesh_line.line_length; i++) {
            int idx = g_mesh_line.line_order[i];
            const char *marker = "";
            if (g_mesh_routing.nodes[idx].node_id == g_mesh_line.head.node_id) {
                marker = " [HEAD]";
            } else if (g_mesh_routing.nodes[idx].node_id == g_mesh_line.tail.node_id) {
                marker = " [TAIL]";
            } else if (g_mesh_routing.nodes[idx].node_id == g_mesh_routing.local_node_id) {
                marker = " [US]";
            }
            MESH_INFO("    [%d] Node %d: 0x%08x%s", i, idx,
                      g_mesh_routing.nodes[idx].node_id, marker);
        }
    }
}

/*
 * =============================================================================
 * Topology-Aware Routing Optimization
 * =============================================================================
 */

/*
 * Initialize topology-specific optimizations
 * Should be called after mesh_build_routing_table() and mesh_detect_topology()
 */
int mesh_topo_optimize_init(void) {
    int ret = 0;

    MESH_INFO("Initializing topology-specific optimizations");

    switch (g_mesh_routing.fast_topology) {
        case MESH_TOPO_RING:
            ret = mesh_ring_init();
            if (ret == 0) {
                mesh_ring_dump_state();
            }
            break;

        case MESH_TOPO_LINE:
            ret = mesh_line_init();
            if (ret == 0) {
                mesh_line_dump_state();
            }
            break;

        case MESH_TOPO_FULL_MESH:
            MESH_INFO("Full mesh topology - no additional optimizations needed");
            break;

        default:
            MESH_INFO("Generic topology - using standard BFS routing");
            break;
    }

    return ret;
}

/*
 * Get optimized route considering topology
 * Enhances the base route with topology-specific information
 */
int mesh_topo_get_optimized_route(uint32_t dest_node_id, struct mesh_route_entry *route) {
    if (!route) {
        return -1;
    }

    /* Start with base route */
    if (mesh_get_route(dest_node_id, route) != 0) {
        return -1;
    }

    /* Apply topology-specific optimizations */
    switch (g_mesh_routing.fast_topology) {
        case MESH_TOPO_RING:
            if (g_mesh_ring.is_ring && !route->is_direct) {
                /* For ring, select path based on load balancing */
                enum mesh_ring_direction dir = mesh_ring_select_path(dest_node_id, 0);
                struct mesh_ring_dual_path *dp = mesh_ring_get_dual_path(dest_node_id);

                if (dp && dp->is_valid && dir != MESH_RING_DIR_NONE) {
                    /* Update route with selected path */
                    if (dir == MESH_RING_DIR_CW) {
                        route->next_hop_node_id = dp->cw_next_hop_id;
                        route->next_hop_ip = dp->cw_next_hop_ip;
                        route->next_hop_nic_idx = dp->cw_next_hop_nic;
                        route->num_hops = dp->cw_path_len;
                    } else {
                        route->next_hop_node_id = dp->ccw_next_hop_id;
                        route->next_hop_ip = dp->ccw_next_hop_ip;
                        route->next_hop_nic_idx = dp->ccw_next_hop_nic;
                        route->num_hops = dp->ccw_path_len;
                    }

                    MESH_DEBUG("Ring optimized route to 0x%08x: %s path, %d hops",
                               dest_node_id, dir == MESH_RING_DIR_CW ? "CW" : "CCW",
                               route->num_hops);
                }
            }
            break;

        case MESH_TOPO_LINE:
            if (g_mesh_line.is_line) {
                /* For line, use direction-based routing */
                uint32_t next_hop = mesh_line_get_next_hop(dest_node_id);
                if (next_hop != 0) {
                    route->next_hop_node_id = next_hop;
                    route->num_hops = mesh_line_hop_count(dest_node_id);

                    MESH_DEBUG("Line optimized route to 0x%08x: direction=%d, %d hops",
                               dest_node_id, mesh_line_get_direction(dest_node_id),
                               route->num_hops);
                }
            }
            break;

        default:
            /* Use base route as-is */
            break;
    }

    return 0;
}

/*
 * Select best path for a message (considers load balancing for ring)
 * Returns the next hop information for sending
 */
int mesh_topo_select_path(uint32_t dest_node_id, size_t msg_size,
                          uint32_t *next_hop_id, uint32_t *next_hop_ip) {
    if (!next_hop_id || !next_hop_ip) {
        return -1;
    }

    *next_hop_id = 0;
    *next_hop_ip = 0;

    /* Check if destination is directly reachable */
    if (mesh_has_direct_route(dest_node_id)) {
        struct mesh_route_entry route;
        if (mesh_get_route(dest_node_id, &route) == 0) {
            *next_hop_id = dest_node_id;
            *next_hop_ip = route.direct_ip;
            return 0;
        }
    }

    /* Apply topology-specific path selection */
    switch (g_mesh_routing.fast_topology) {
        case MESH_TOPO_RING:
            if (g_mesh_ring.is_ring) {
                enum mesh_ring_direction dir = mesh_ring_select_path(dest_node_id, msg_size);
                struct mesh_ring_dual_path *dp = mesh_ring_get_dual_path(dest_node_id);

                if (dp && dp->is_valid && dir != MESH_RING_DIR_NONE) {
                    if (dir == MESH_RING_DIR_CW) {
                        *next_hop_id = dp->cw_next_hop_id;
                        *next_hop_ip = dp->cw_next_hop_ip;
                    } else {
                        *next_hop_id = dp->ccw_next_hop_id;
                        *next_hop_ip = dp->ccw_next_hop_ip;
                    }

                    /* Update statistics for load balancing */
                    mesh_ring_update_stats(dest_node_id, dir, msg_size);

                    MESH_DEBUG("Ring path selected: dest=0x%08x, dir=%s, next_hop=0x%08x",
                               dest_node_id, dir == MESH_RING_DIR_CW ? "CW" : "CCW",
                               *next_hop_id);
                    return 0;
                }
            }
            break;

        case MESH_TOPO_LINE:
            if (g_mesh_line.is_line) {
                *next_hop_id = mesh_line_get_next_hop(dest_node_id);
                if (*next_hop_id != 0) {
                    /* Find IP for next hop from adjacencies */
                    for (int i = 0; i < g_mesh_routing.num_adjacencies; i++) {
                        if (g_mesh_routing.adjacencies[i].node_id == *next_hop_id &&
                            g_mesh_routing.adjacencies[i].is_fast_adjacent) {
                            uint32_t shared = g_mesh_routing.adjacencies[i].shared_fast_subnet;
                            int next_idx = mesh_get_node_index(*next_hop_id);
                            if (next_idx >= 0) {
                                struct mesh_node_identity *next = &g_mesh_routing.nodes[next_idx];
                                for (int a = 0; a < next->num_fast_addrs; a++) {
                                    uint32_t peer_ip = ntohl(next->fast_addrs[a].ip);
                                    uint32_t peer_mask = ntohl(next->fast_addrs[a].mask);
                                    if ((peer_ip & peer_mask) == shared) {
                                        *next_hop_ip = peer_ip;
                                        break;
                                    }
                                }
                            }
                            break;
                        }
                    }

                    MESH_DEBUG("Line path selected: dest=0x%08x, next_hop=0x%08x",
                               dest_node_id, *next_hop_id);
                    return 0;
                }
            }
            break;

        default:
            break;
    }

    /* Fall back to standard routing */
    struct mesh_route_entry route;
    if (mesh_get_route(dest_node_id, &route) == 0 && route.reachable) {
        if (route.is_direct) {
            *next_hop_id = dest_node_id;
            *next_hop_ip = route.direct_ip;
        } else {
            *next_hop_id = route.next_hop_node_id;
            *next_hop_ip = route.next_hop_ip;
        }
        return 0;
    }

    MESH_WARN("No path to destination 0x%08x", dest_node_id);
    return -1;
}
