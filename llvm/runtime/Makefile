# Libraries Makefile:  Build all subdirectories automatically

all:: runtime.o

LEVEL = ../..
DIRS  := $(sort $(filter-out Output/, $(filter-out CVS/, $(wildcard */))))

include ${LEVEL}/Makefile.common

# Install target for libraries: Copy into the gcc install directory in chris's
# tree...
#
install::

clean::

runtime.o: ../runtime.c
	$(CC) -c $(CFLAGS) $<
