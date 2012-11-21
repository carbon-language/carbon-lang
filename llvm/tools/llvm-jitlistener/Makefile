##===- tools/llvm-jitlistener/Makefile ---------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL := ../..
TOOLNAME := llvm-jitlistener

include $(LEVEL)/Makefile.config

LINK_COMPONENTS := mcjit jit interpreter nativecodegen bitreader asmparser selectiondag Object

# If Intel JIT Events support is configured, link against the LLVM Intel JIT
# Events interface library.  If not, this tool will do nothing useful, but it
# will build correctly.
ifeq ($(USE_INTEL_JITEVENTS), 1)
  LINK_COMPONENTS += inteljitevents
endif

# This tool has no plugins, optimize startup time.
TOOL_NO_EXPORTS := 1

include $(LLVM_SRC_ROOT)/Makefile.rules
