##===- tools/llc/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = llc
USEDLIBS = sparc \
           x86 \
	   selectiondag \
           regalloc \
           sched \
           select \
           codegen \
           target.a \
           livevar \
           transforms.a \
           scalaropts.a \
           analysis.a \
           transformutils.a \
           bcreader \
           bcwriter \
           vmcore \
           support.a
TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common

