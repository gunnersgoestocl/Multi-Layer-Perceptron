CC = gcc # C compiler
AR = ar # archiver
TODAY   = $(shell date "+%Y%m%d%H%M%S")

# Output directories
BIN_DIR = bin
LIB_DIR = lib
SRC_DIR = src
INCLUDE_DIR = include

# Source files and object files
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:.c=.o)

# Target binaries
TARGET = $(BIN_DIR)/demo

# Static library
LIBRARY = $(LIB_DIR)/libadv_regression.a

# Build the demo executable
$(TARGET): $(LIBRARY)
	$(CC) -v -o $(TARGET) -L$(LIB_DIR) -ladv_regression -lm

# Archive the library
$(LIBRARY): $(OBJS)
	$(AR) rsv $@ $(OBJS)

# Compile the object files
%.o: %.c
	$(CC) -I$(INCLUDE_DIR) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(SRC_DIR)/*.o $(LIB_DIR)/*.a $(BIN_DIR)/*