##===- lib/Target/Skeleton/Makefile ------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by the LLVM research group and is distributed under
# the University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMSkeleton

TARGET = Skeleton
# Make sure that tblgen is run, first thing.
BUILT_SOURCES = $(TARGET)GenRegisterInfo.h.inc $(TARGET)GenRegisterNames.inc \
                $(TARGET)GenRegisterInfo.inc $(TARGET)GenInstrNames.inc \
                $(TARGET)GenInstrInfo.inc

include $(LEVEL)/Makefile.common
