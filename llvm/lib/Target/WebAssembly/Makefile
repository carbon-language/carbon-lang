##===- lib/Target/WebAssembly/Makefile ---------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMWebAssemblyCodeGen
TARGET = WebAssembly

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = WebAssemblyGenRegisterInfo.inc WebAssemblyGenSubtargetInfo.inc \
		WebAssemblyGenMCCodeEmitter.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
