##===- tools/edis/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL := ../..
LIBRARYNAME := EnhancedDisassembly

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := mcdisassembler

# If the X86 target is enabled, link in the asmprinter and disassembler.
ifneq ($(filter $(TARGETS_TO_BUILD), X86),)
LINK_COMPONENTS += x86asmprinter x86disassembler
endif

# If the ARM target is enabled, link in the asmprinter and disassembler.
ifneq ($(filter $(TARGETS_TO_BUILD), ARM),)
LINK_COMPONENTS += armasmprinter armdisassembler
endif

include $(LEVEL)/Makefile.common
