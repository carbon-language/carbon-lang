##===- tools/opt/Makefile ------------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
LEVEL = ../..
TOOLNAME = opt

USEDLIBS = bcreader bcwriter \
           instrument profpaths scalaropts \
           ipo ipa.a datastructure transforms target.a analysis \
           transformutils vmcore support
TOOLLINKOPTS = $(PLATFORMLIBDL)

include $(LEVEL)/Makefile.common
