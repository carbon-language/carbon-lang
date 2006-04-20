##===- tools/Makefile --------------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ..
PARALLEL_DIRS := llvm-as llvm-dis opt gccas llc llvm-link lli gccld llvm-stub \
                 analyze llvm-extract llvm-nm llvm-prof llvm-ar llvm-ranlib \
                 llvm-bcanalyzer llvmc llvm-ld llvm-db bugpoint llvm-config

include $(LEVEL)/Makefile.config

# The bugpoint and llvm-db tools are not portable to Win32 because they depend
# on fork(2) behavior that Win32 doesn't have. At some point they'll be 
# rewritten to not depend on fork at which time they should be added back to
# the list above.
ifneq ($(LLVM_ON_UNIX),1)
PARALLEL_DIRS := $(filter-out bugpoint llvm-db,$(PARALLEL_DIRS))
endif

include $(LLVM_SRC_ROOT)/Makefile.rules
