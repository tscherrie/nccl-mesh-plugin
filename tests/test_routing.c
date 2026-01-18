/*
 * NCCL Mesh Plugin - Routing Unit Tests
 *
 * Tests for topology discovery, routing table construction, and path computation.
 *
 * Build: make test_routing
 * Run:   ./test_routing
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <arpa/inet.h>
#include <pthread.h>

/*
 * For test builds, we include the real headers but stub out
 * functions that require RDMA/libibverbs
 */

/* Include the real headers */
#include "mesh_plugin.h"
#include "mesh_routing.h"

/* Global state instance for testing */
struct mesh_plugin_state g_mesh_state = {0};

/*
 * Stub functions - these are defined in mesh_plugin.c but we need stubs
 * for the test executable to link without pulling in all of mesh_plugin.c
 */

/* Convert uint32_t IP to string (stub) */
void mesh_uint_to_ip(uint32_t ip, char *buf, size_t len) {
    struct in_addr addr;
    addr.s_addr = htonl(ip);
    snprintf(buf, len, "%s", inet_ntoa(addr));
}

/* Find NIC for IP (stub - returns NULL for tests) */
struct mesh_nic* mesh_find_nic_for_ip(uint32_t peer_ip) {
    (void)peer_ip;
    return NULL;
}

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/*
 * Test framework macros
 */
#define TEST_START(name) \
    do { \
        tests_run++; \
        printf("TEST: %s... ", name); \
        fflush(stdout); \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("PASS\n"); \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        tests_failed++; \
        printf("FAIL: %s\n", msg); \
    } while(0)

#define ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_NE(a, b, msg) \
    do { \
        if ((a) == (b)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            TEST_FAIL(msg); \
            return; \
        } \
    } while(0)

/*
 * =============================================================================
 * Test: NIC Lane Classification
 * =============================================================================
 */

static void test_nic_lane_classification(void) {
    TEST_START("NIC lane classification");

    /* 10 Gbps - management */
    ASSERT_EQ(mesh_classify_nic_lane(10000), MESH_LANE_MANAGEMENT,
              "10 Gbps should be management");

    /* 25 Gbps - management */
    ASSERT_EQ(mesh_classify_nic_lane(25000), MESH_LANE_MANAGEMENT,
              "25 Gbps should be management");

    /* 100 Gbps - fast */
    ASSERT_EQ(mesh_classify_nic_lane(100000), MESH_LANE_FAST,
              "100 Gbps should be fast lane");

    /* 200 Gbps - fast */
    ASSERT_EQ(mesh_classify_nic_lane(200000), MESH_LANE_FAST,
              "200 Gbps should be fast lane");

    /* 50 Gbps - fast (at threshold) */
    ASSERT_EQ(mesh_classify_nic_lane(50000), MESH_LANE_FAST,
              "50 Gbps should be fast lane");

    /* Unknown/invalid */
    ASSERT_EQ(mesh_classify_nic_lane(0), MESH_LANE_UNKNOWN,
              "0 should be unknown");
    ASSERT_EQ(mesh_classify_nic_lane(-1), MESH_LANE_UNKNOWN,
              "-1 should be unknown");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Lane Name Strings
 * =============================================================================
 */

static void test_lane_names(void) {
    TEST_START("Lane name strings");

    ASSERT_TRUE(strcmp(mesh_lane_name(MESH_LANE_UNKNOWN), "unknown") == 0,
                "UNKNOWN lane name");
    ASSERT_TRUE(strcmp(mesh_lane_name(MESH_LANE_MANAGEMENT), "management") == 0,
                "MANAGEMENT lane name");
    ASSERT_TRUE(strcmp(mesh_lane_name(MESH_LANE_FAST), "fast") == 0,
                "FAST lane name");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Topology Name Strings
 * =============================================================================
 */

static void test_topology_names(void) {
    TEST_START("Topology name strings");

    ASSERT_TRUE(strcmp(mesh_topology_name(MESH_TOPO_UNKNOWN), "unknown") == 0,
                "UNKNOWN topology name");
    ASSERT_TRUE(strcmp(mesh_topology_name(MESH_TOPO_FULL_MESH), "full_mesh") == 0,
                "FULL_MESH topology name");
    ASSERT_TRUE(strcmp(mesh_topology_name(MESH_TOPO_RING), "ring") == 0,
                "RING topology name");
    ASSERT_TRUE(strcmp(mesh_topology_name(MESH_TOPO_LINE), "line") == 0,
                "LINE topology name");
    ASSERT_TRUE(strcmp(mesh_topology_name(MESH_TOPO_STAR), "star") == 0,
                "STAR topology name");
    ASSERT_TRUE(strcmp(mesh_topology_name(MESH_TOPO_PARTIAL), "partial") == 0,
                "PARTIAL topology name");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Node ID Generation
 * =============================================================================
 */

static void test_node_id_generation(void) {
    TEST_START("Node ID generation");

    /* Set up mock NIC state */
    g_mesh_state.num_nics = 2;
    g_mesh_state.nics[0].ip_addr = inet_addr("192.168.1.10");
    g_mesh_state.nics[1].ip_addr = inet_addr("192.168.2.10");

    uint32_t node_id = mesh_generate_node_id();

    /* Node ID should be non-zero */
    ASSERT_NE(node_id, 0, "Node ID should be non-zero");

    /* Same IPs should generate same ID */
    uint32_t node_id2 = mesh_generate_node_id();
    ASSERT_EQ(node_id, node_id2, "Same IPs should generate same node ID");

    /* Different IPs should generate different ID */
    g_mesh_state.nics[0].ip_addr = inet_addr("192.168.1.20");
    uint32_t node_id3 = mesh_generate_node_id();
    ASSERT_NE(node_id, node_id3, "Different IPs should generate different node ID");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Ring Direction Selection
 * =============================================================================
 */

static void test_ring_direction(void) {
    TEST_START("Ring direction selection");

    /* Test direction enum values */
    ASSERT_EQ(MESH_RING_DIR_NONE, 0, "DIR_NONE should be 0");
    ASSERT_EQ(MESH_RING_DIR_CW, 1, "DIR_CW should be 1");
    ASSERT_EQ(MESH_RING_DIR_CCW, 2, "DIR_CCW should be 2");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Ring Hop Count Calculation
 * =============================================================================
 */

static void test_ring_hop_count_math(void) {
    TEST_START("Ring hop count math");

    /* For an N-node ring, CW + CCW distance always equals N */
    /* Test with a simulated 4-node ring: positions 0, 1, 2, 3 */
    int n = 4;
    int our_pos = 0;

    /* Destination at position 1 */
    int dest_pos = 1;
    int cw_dist = (dest_pos - our_pos + n) % n;   /* 1 */
    int ccw_dist = (our_pos - dest_pos + n) % n;  /* 3 */
    ASSERT_EQ(cw_dist, 1, "CW to pos 1 should be 1");
    ASSERT_EQ(ccw_dist, 3, "CCW to pos 1 should be 3");
    ASSERT_EQ(cw_dist + ccw_dist, n, "CW + CCW should equal N");

    /* Destination at position 2 (opposite) */
    dest_pos = 2;
    cw_dist = (dest_pos - our_pos + n) % n;   /* 2 */
    ccw_dist = (our_pos - dest_pos + n) % n;  /* 2 */
    ASSERT_EQ(cw_dist, 2, "CW to pos 2 should be 2");
    ASSERT_EQ(ccw_dist, 2, "CCW to pos 2 should be 2");

    /* Destination at position 3 */
    dest_pos = 3;
    cw_dist = (dest_pos - our_pos + n) % n;   /* 3 */
    ccw_dist = (our_pos - dest_pos + n) % n;  /* 1 */
    ASSERT_EQ(cw_dist, 3, "CW to pos 3 should be 3");
    ASSERT_EQ(ccw_dist, 1, "CCW to pos 3 should be 1");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Line Position Math
 * =============================================================================
 */

static void test_line_position_math(void) {
    TEST_START("Line position math");

    /* For a line: A-B-C-D (positions 0,1,2,3) */
    /* From B (pos 1) to D (pos 3): direction = tail (+1), hops = 2 */

    int our_pos = 1;  /* B */
    int dest_pos = 3; /* D */

    int direction = (dest_pos > our_pos) ? 1 : ((dest_pos < our_pos) ? -1 : 0);
    int hops = (dest_pos > our_pos) ? (dest_pos - our_pos) : (our_pos - dest_pos);

    ASSERT_EQ(direction, 1, "B->D should be towards tail (+1)");
    ASSERT_EQ(hops, 2, "B->D should be 2 hops");

    /* From B (pos 1) to A (pos 0): direction = head (-1), hops = 1 */
    dest_pos = 0;
    direction = (dest_pos > our_pos) ? 1 : ((dest_pos < our_pos) ? -1 : 0);
    hops = (dest_pos > our_pos) ? (dest_pos - our_pos) : (our_pos - dest_pos);

    ASSERT_EQ(direction, -1, "B->A should be towards head (-1)");
    ASSERT_EQ(hops, 1, "B->A should be 1 hop");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Relay Session ID Generation
 * =============================================================================
 */

static void test_relay_session_id(void) {
    TEST_START("Relay session ID generation");

    /* Initialize relay state for testing */
    memset(&g_mesh_relay, 0, sizeof(g_mesh_relay));
    pthread_mutex_init(&g_mesh_relay.sessions_mutex, NULL);
    g_mesh_relay.next_session_id = 0x12340000;

    uint32_t id1 = mesh_relay_generate_session_id();
    uint32_t id2 = mesh_relay_generate_session_id();
    uint32_t id3 = mesh_relay_generate_session_id();

    /* IDs should be sequential */
    ASSERT_EQ(id1, 0x12340000, "First ID should match seed");
    ASSERT_EQ(id2, 0x12340001, "Second ID should be seed+1");
    ASSERT_EQ(id3, 0x12340002, "Third ID should be seed+2");

    pthread_mutex_destroy(&g_mesh_relay.sessions_mutex);

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Relay Header Magic
 * =============================================================================
 */

static void test_relay_header_magic(void) {
    TEST_START("Relay header magic");

    ASSERT_EQ(MESH_RELAY_MAGIC, 0x52454C59, "RELAY magic should be 'RELY'");
    ASSERT_EQ(MESH_NODE_ID_MAGIC, 0x4E4F4445, "NODE magic should be 'NODE'");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Route Entry Structure
 * =============================================================================
 */

static void test_route_entry_structure(void) {
    TEST_START("Route entry structure");

    struct mesh_route_entry route;
    memset(&route, 0, sizeof(route));

    /* Test structure field assignments */
    route.dest_node_id = 0x12345678;
    route.dest_node_idx = 5;
    route.reachable = 1;
    route.num_hops = 3;
    route.is_direct = 0;
    route.next_hop_node_id = 0xAABBCCDD;

    ASSERT_EQ(route.dest_node_id, 0x12345678, "dest_node_id");
    ASSERT_EQ(route.dest_node_idx, 5, "dest_node_idx");
    ASSERT_EQ(route.reachable, 1, "reachable");
    ASSERT_EQ(route.num_hops, 3, "num_hops");
    ASSERT_EQ(route.is_direct, 0, "is_direct");
    ASSERT_EQ(route.next_hop_node_id, 0xAABBCCDD, "next_hop_node_id");

    /* Test relay path */
    route.path_len = 4;
    route.relay_path[0] = 0;
    route.relay_path[1] = 1;
    route.relay_path[2] = 2;
    route.relay_path[3] = 3;

    ASSERT_EQ(route.path_len, 4, "path_len");
    ASSERT_EQ(route.relay_path[0], 0, "relay_path[0]");
    ASSERT_EQ(route.relay_path[3], 3, "relay_path[3]");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Ring Dual Path Structure
 * =============================================================================
 */

static void test_ring_dual_path_structure(void) {
    TEST_START("Ring dual path structure");

    struct mesh_ring_dual_path dp;
    memset(&dp, 0, sizeof(dp));

    dp.dest_node_id = 0x11223344;
    dp.is_valid = 1;
    dp.cw_path_len = 2;
    dp.ccw_path_len = 3;
    dp.cw_next_hop_id = 0xAAAAAAAA;
    dp.ccw_next_hop_id = 0xBBBBBBBB;
    dp.cw_bytes_sent = 1000000;
    dp.ccw_bytes_sent = 2000000;
    dp.preferred = MESH_RING_DIR_CW;

    ASSERT_EQ(dp.dest_node_id, 0x11223344, "dest_node_id");
    ASSERT_EQ(dp.is_valid, 1, "is_valid");
    ASSERT_EQ(dp.cw_path_len, 2, "cw_path_len");
    ASSERT_EQ(dp.ccw_path_len, 3, "ccw_path_len");
    ASSERT_EQ(dp.cw_next_hop_id, 0xAAAAAAAA, "cw_next_hop_id");
    ASSERT_EQ(dp.ccw_next_hop_id, 0xBBBBBBBB, "ccw_next_hop_id");
    ASSERT_EQ(dp.cw_bytes_sent, 1000000UL, "cw_bytes_sent");
    ASSERT_EQ(dp.ccw_bytes_sent, 2000000UL, "ccw_bytes_sent");
    ASSERT_EQ(dp.preferred, MESH_RING_DIR_CW, "preferred");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Line Endpoint Structure
 * =============================================================================
 */

static void test_line_endpoint_structure(void) {
    TEST_START("Line endpoint structure");

    struct mesh_line_endpoint ep;
    memset(&ep, 0, sizeof(ep));

    ep.node_id = 0x55667788;
    ep.node_idx = 3;
    ep.is_head = 1;
    ep.is_tail = 0;
    ep.neighbor_id = 0x99AABBCC;

    ASSERT_EQ(ep.node_id, 0x55667788, "node_id");
    ASSERT_EQ(ep.node_idx, 3, "node_idx");
    ASSERT_EQ(ep.is_head, 1, "is_head");
    ASSERT_EQ(ep.is_tail, 0, "is_tail");
    ASSERT_EQ(ep.neighbor_id, 0x99AABBCC, "neighbor_id");

    TEST_PASS();
}

/*
 * =============================================================================
 * Test: Constants and Limits
 * =============================================================================
 */

static void test_constants_and_limits(void) {
    TEST_START("Constants and limits");

    /* Check that limits are reasonable */
    ASSERT_TRUE(MESH_MAX_NODES >= 4, "MAX_NODES should be at least 4");
    ASSERT_TRUE(MESH_MAX_NODES <= 256, "MAX_NODES should be at most 256");

    ASSERT_TRUE(MESH_MAX_HOPS >= 2, "MAX_HOPS should be at least 2");
    ASSERT_TRUE(MESH_MAX_HOPS <= 16, "MAX_HOPS should be at most 16");

    ASSERT_EQ(MESH_INVALID_NODE, 0xFF, "INVALID_NODE should be 0xFF");

    /* Speed thresholds */
    ASSERT_TRUE(MESH_SPEED_FAST_LANE_MIN >= 40000, "Fast lane min should be >= 40 Gbps");
    ASSERT_TRUE(MESH_SPEED_MANAGEMENT_MAX <= 30000, "Management max should be <= 30 Gbps");
    ASSERT_TRUE(MESH_SPEED_FAST_LANE_MIN > MESH_SPEED_MANAGEMENT_MAX,
                "Fast lane min should be > management max");

    /* Relay buffer size should be reasonable */
    ASSERT_TRUE(MESH_RELAY_BUFFER_SIZE >= 1024*1024, "Relay buffer should be >= 1MB");
    ASSERT_TRUE(MESH_RELAY_BUFFER_SIZE <= 64*1024*1024, "Relay buffer should be <= 64MB");

    TEST_PASS();
}

/*
 * =============================================================================
 * Main Test Runner
 * =============================================================================
 */

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("\n");
    printf("=====================================================\n");
    printf("  NCCL Mesh Plugin - Routing Unit Tests\n");
    printf("=====================================================\n\n");

    /* Initialize minimal state */
    g_mesh_state.debug_level = 0;  /* Quiet mode for tests */

    /* Run all tests */
    test_nic_lane_classification();
    test_lane_names();
    test_topology_names();
    test_node_id_generation();
    test_ring_direction();
    test_ring_hop_count_math();
    test_line_position_math();
    test_relay_session_id();
    test_relay_header_magic();
    test_route_entry_structure();
    test_ring_dual_path_structure();
    test_line_endpoint_structure();
    test_constants_and_limits();

    /* Print summary */
    printf("\n=====================================================\n");
    printf("  Results: %d tests run, %d passed, %d failed\n",
           tests_run, tests_passed, tests_failed);
    printf("=====================================================\n\n");

    return (tests_failed > 0) ? 1 : 0;
}
