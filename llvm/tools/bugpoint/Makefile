##===- tools/bugpoint/Makefile -----------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..

TOOLNAME = bugpoint

OPTLIBS  = transforms instrument profpaths 
ANALIBS  = datastructure ipa target.a 

USEDLIBS = ipo scalaropts analysis $(OPTLIBS) $(ANALIBS) \
           transformutils asmparser bcreader bcwriter vmcore support

TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common
