CC=g++
NVCC = nvcc
CFLAGS= -O3 -Xcompiler -fopenmp

BUILD_DIR=build
SRC_DIR=src
INCLUDE_DIR=./include
DATA_DIR=data/out
SOURCES := $(shell find $(SRC_DIR) -name '*.cu')

$(info $(shell mkdir -p $(BUILD_DIR)))
$(info $(shell mkdir -p $(DATA_DIR)))

default: compile

compile:
	$(NVCC) -o $(BUILD_DIR)/main -I$(INCLUDE_DIR) $(SOURCES) $(CFLAGS) 

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR)