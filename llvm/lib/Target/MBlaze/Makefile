##===- lib/Target/MBlaze/Makefile --------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##
LEVEL = ../../..
LIBRARYNAME = LLVMMBlazeCodeGen
TARGET = MBlaze

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = MBlazeGenRegisterInfo.h.inc MBlazeGenRegisterNames.inc \
                MBlazeGenRegisterInfo.inc MBlazeGenInstrNames.inc \
                MBlazeGenInstrInfo.inc MBlazeGenAsmWriter.inc \
                MBlazeGenDAGISel.inc MBlazeGenCallingConv.inc \
                MBlazeGenSubtarget.inc MBlazeGenIntrinsics.inc

DIRS = AsmPrinter TargetInfo

include $(LEVEL)/Makefile.common

