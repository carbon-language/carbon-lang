##===------- lld/Makefile --------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===--------------------------------------------------------------------===##

LLD_LEVEL := ../..

TOOLNAME = lld

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

#include /Makefile.config
LEVEL := $(LLD_LEVEL)/../..
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD)
USEDLIBS = lldDriver.a lldConfig.a \
           lldELF.a lldMachO.a lldPasses.a lldPECOFF.a lldYAML.a \
           lldReaderWriter.a lldCore.a lldNative.a \
           lldHexagonELFTarget.a lldPPCELFTarget.a lldMipsELFTarget.a \
           lldX86ELFTarget.a lldX86_64ELFTarget.a lldAArch64ELFTarget.a \
           LLVMOption.a

include $(LLD_LEVEL)/Makefile
