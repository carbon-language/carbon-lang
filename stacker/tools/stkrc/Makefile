#===-- projects/llvm-stacker/tools/stkrc/Makefile -----------*- Makefile -*-===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

# Indicate where we are relative to the top of the source tree.
LEVEL=../..

# Give the name of a library.  This will build a dynamic version.
TOOLNAME = stkrc

# Define the link components
USEDLIBS=stkr_compiler 
LINK_COMPONENTS = asmparser bcwriter ipo scalaropts \
		  transformutils ipa analysis target $(TARGETS_TO_BUILD) core

# Define our configuration files
CONFIG_FILES = st
EXTRA_DIST = st

# We need exception handling
REQUIRES_EH := 1

ifdef PARSE_DEBUG
CPPFLAGS = -DPARSE_DEBUG=1
endif

ifdef FLEX_DEBUG
CPPFLAGS += -DFLEX_DEBUG=1
endif

include $(LEVEL)/Makefile.common
