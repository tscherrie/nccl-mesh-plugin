# NCCL Mesh Plugin Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -g
CFLAGS += -I. -I./include
# LDFLAGS for systems with libibverbs
# LDFLAGS = -shared -libverbs -lpthread
# LDFLAGS for stub/TCP-only build
LDFLAGS = -shared -lpthread

# Target
TARGET = libnccl-net.so
TARGET_MESH = libnccl-net-mesh.so

# Sources
SRCS = src/mesh_plugin.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET) $(TARGET_MESH)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

$(TARGET_MESH): $(TARGET)
	ln -sf $(TARGET) $(TARGET_MESH)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Install to a standard location
PREFIX ?= /usr/local
install: all
	install -d $(PREFIX)/lib
	install -m 755 $(TARGET) $(PREFIX)/lib/
	ln -sf $(TARGET) $(PREFIX)/lib/$(TARGET_MESH)

# Clean
clean:
	rm -f $(OBJS) $(TARGET) $(TARGET_MESH)

# Test build (requires libibverbs-dev)
test-deps:
	@echo "Checking dependencies..."
	@pkg-config --exists libibverbs || (echo "ERROR: libibverbs-dev not found" && exit 1)
	@echo "All dependencies found."

# Debug build
debug: CFLAGS += -DDEBUG -g3 -O0
debug: clean all

# Print configuration
info:
	@echo "CC      = $(CC)"
	@echo "CFLAGS  = $(CFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"
	@echo "TARGET  = $(TARGET)"

.PHONY: all clean install test-deps debug info
