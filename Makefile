c_files = $(wildcard *.c)
o_files = $(patsubst %.c, %.o, $(c_files))
d_files = $(patsubst %.c, %.d, $(c_files))

CURRENT_DIR := $(shell pwd)

CC = gcc
INCLUDE_DIR := -I$(CURRENT_DIR)
CC_FLAGS = -Wall -O3 -g $(addprefix -I, $(INCLUDE_DIR)) 
LDFLAGS = -lm
CC_DEPFLAGS	=-MMD -MF $(@:.o=.d) -MT $@
TARGET = micrograd

all: print_c print_o $(TARGET)

print_c:
	echo $(c_files)

print_o:
	echo $(o_files)

%.o: %.c
	$(CC) $(CC_FLAGS) $(CC_DEPFLAGS) -c $< -o $@

$(TARGET): $(o_files)
	$(CC) $(CC_FLAGS) -o $(TARGET) $(o_files) $(LDFLAGS)

clean:
	rm -f $(o_files) $(d_files) $(TARGET)

-include $(wildcard $(o_files:.o=.d))

.PHONY: all print_c print_o clean
