#===- tools/llc/Makefile -----------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../..
TOOLNAME = llc
USEDLIBS = \
	cwriter \
	powerpc \
	sparcv9 \
	x86 \
	skeleton \
	selectiondag \
	sparcv9regalloc \
	sched \
	codegen \
	target.a \
	sparcv9livevar \
	ipa.a \
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

