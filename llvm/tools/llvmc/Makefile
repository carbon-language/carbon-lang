##===- tools/llvmc/Makefile --------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = llvmc
USEDLIBS = LLVMBCReader LLVMCore LLVMSupport.a LLVMbzip2 LLVMSystem.a
CONFIG_FILES = c cpp ll st 
EXTRA_DIST = c cpp ll st

include $(LEVEL)/Makefile.common

install::
	$(Echo) Installing additional C++ configuration clones
	$(Verb)$(DataInstall) $(PROJ_OBJ_DIR)/cpp $(PROJ_etcdir)/cc
	$(Verb)$(DataInstall) $(PROJ_OBJ_DIR)/cpp $(PROJ_etcdir)/c++
	$(Verb)$(DataInstall) $(PROJ_OBJ_DIR)/cpp $(PROJ_etcdir)/cxx
	$(Verb)$(DataInstall) $(PROJ_OBJ_DIR)/cpp $(PROJ_etcdir)/C
