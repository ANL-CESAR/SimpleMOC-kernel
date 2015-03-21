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

include ${OCCA_DIR}/scripts/makefile

program = SimpleMOC-kernel

source = \
main.c \
init.c \
io.c 

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Regular gcc Compiler
ifeq ($(COMPILER),gcc)
  CC = gcc
  CXX = g++
endif

# Standard Flags
CFLAGS := -std=gnu99 -Wall -I${OCCA_DIR}/include

# Linker Flags
LDFLAGS = -lm -L${OCCA_DIR}/lib -locca

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
	$(CXX) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) *.okl SimpleMOC-kernel_header.h

run:
	./$(program)
