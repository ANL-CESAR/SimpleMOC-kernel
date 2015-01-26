#===============================================================================
# User Options
#===============================================================================

COMPILER    = gcc
OPTIMIZE    = no
DEBUG       = no
PROFILE     = no
BENCHMARK   = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = SimpleMOC-kernel

source = \
init.c \
io.c \
main.c \

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Regular gcc Compiler
ifeq ($(COMPILER),gcc)
  CC = gcc
endif

# Standard Flags
CFLAGS := -std=gnu99 -Wall -I${OCCA_DIR}/include

# Linker Flags
LDFLAGS = -lm -L${OCCA_DIR} -locca

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3 --use_fast_math
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) SimpleMOC-kernel_header.h
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) SimpleMOC-kernel_header.h

run:
	./$(program)
