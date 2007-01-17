##===- projects/sample/Makefile ----------------------------*- Makefile -*-===##
#
# This is a sample Makefile for a project that uses LLVM.
#
##===----------------------------------------------------------------------===##

#
# Indicates our relative path to the top of the project's root directory.
#
LEVEL = ../../..
DIRS = 

EXTRA_DIST = fibonacci.st hello.st prime.st goof.st

SAMPLES = fibonacci hello prime goof

LLVMC_EXEC = LLVM_CONFIG_DIR=$(BUILD_SRC_ROOT)/tools/llvmc $(TOOLDIR)/llvmc

all :: $(SAMPLES)

ifdef OPTIMIZE
% : %.st 
	$(Echo) "Compiling and Optimizing $(<F)"
	$(Verb)$(LLVMC_EXEC) -O3 $< -o $@
else
% : %.st
	$(Echo) "Compiling $(<F)"
	$(Verb)$(LLVMC_EXEC) $< -o $@
endif

SAMPLES_LL = $(SAMPLES:%=%.ll)
SAMPLES_BC = $(SAMPLES:%=%.bc)
SAMPLES_S  = $(SAMPLES:%=%.s)

clean ::
	$(Verb)rm -f gmon.out $(SAMPLES)
#
# Include the Master Makefile that knows how to build all.
#
include $(LEVEL)/Makefile.common
