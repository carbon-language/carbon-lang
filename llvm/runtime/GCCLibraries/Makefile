# llvm/runtime/GCCLibraries Makefile:  Build all subdirectories automatically

LEVEL = ../..
PARALLEL_DIRS  := $(sort $(filter-out Output/, $(filter-out CVS/, $(wildcard */))))
PARALLEL_DIRS := $(filter-out libexception/, $(PARALLEL_DIRS))

include $(LEVEL)/Makefile.common

# Install target for libraries: Copy into the gcc install directory in chris's
# tree...
#
install::

clean::
