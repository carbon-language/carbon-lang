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
USEDLIBS = LLVMBCReader.a LLVMCore.a LLVMSupport.a LLVMbzip2.a LLVMSystem.a
CONFIG_FILES = c cpp ll
EXTRA_DIST = c cpp ll ConfigLexer.cpp.cvs ConfigLexer.l.cvs

# The CompilerDriver needs to know the locations of several configured
# directories and paths. We define these as preprocessor symbols so they can 
# be hard coded into the process based on the configuration. Only those
# configuration values not available in llvm/include/Config/config.h need to be
# specified here. These values are used as the replacements for the
# configuration file substitution variables such as %llvmgccdir%
CPPFLAGS = -DLLVMGCCDIR="\"$(LLVMGCCDIR)\"" \
           -DLLVMGCCARCH="\"$(LLVMGCCARCH)\"" \
	   -DLLVMGCC="\"$(LLVMGCC)\"" \
	   -DLLVMGXX="\"$(LLVMGXX)\"" \
	   -DLLVMCC1="\"$(LLVMCC1)\"" \
	   -DLLVMCC1PLUS="$(LLVMCC1PLUS)"

include $(LEVEL)/Makefile.common

install::
	$(Echo) Installing additional C++ configuration clones
	$(Verb)$(DataInstall) $(PROJ_SRC_DIR)/cpp $(PROJ_etcdir)/c++
	$(Verb)$(DataInstall) $(PROJ_SRC_DIR)/cpp $(PROJ_etcdir)/cxx
