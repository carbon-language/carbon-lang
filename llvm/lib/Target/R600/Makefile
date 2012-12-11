##===- lib/Target/R600/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMR600CodeGen
TARGET = AMDGPU

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = AMDGPUGenRegisterInfo.inc AMDGPUGenInstrInfo.inc \
		AMDGPUGenDAGISel.inc  AMDGPUGenSubtargetInfo.inc \
		AMDGPUGenMCCodeEmitter.inc AMDGPUGenCallingConv.inc \
		AMDGPUGenIntrinsics.inc AMDGPUGenDFAPacketizer.inc \
		AMDGPUGenAsmWriter.inc

DIRS = InstPrinter TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
