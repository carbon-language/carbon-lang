##===- utils/llvm-config/Makefile --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and Eric Kidd and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..


include $(LEVEL)/Makefile.common

LibDeps.txt: $(LEVEL)/utils/GenLibDeps.pl $(LibDir)
	$(LEVEL)/utils/GenLibDeps.pl -flat $(LibDir) | sort > LibDeps.txt
