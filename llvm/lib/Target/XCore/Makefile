##===- lib/Target/XCore/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMXCoreCodeGen
TARGET = XCore

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = XCoreGenRegisterInfo.h.inc XCoreGenRegisterNames.inc \
                XCoreGenRegisterInfo.inc XCoreGenInstrNames.inc \
                XCoreGenInstrInfo.inc XCoreGenAsmWriter.inc \
                XCoreGenDAGISel.inc XCoreGenCallingConv.inc \
		XCoreGenSubtarget.inc

DIRS = AsmPrinter TargetInfo

include $(LEVEL)/Makefile.common

