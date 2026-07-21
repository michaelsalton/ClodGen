# ClodGen Makefile

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS  ?=

BUILD_DIR := build
SRC_DIR   := src
TARGET    := $(BUILD_DIR)/clodgen

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Default to a release-ish build; `make debug` for debug symbols.
CXXFLAGS += -O2

.PHONY: all debug run clean

all: $(TARGET)

debug: CXXFLAGS := $(filter-out -O2,$(CXXFLAGS)) -O0 -g
debug: clean $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
