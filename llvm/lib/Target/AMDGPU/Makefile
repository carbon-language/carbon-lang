##===- lib/Target/AMDGPU/Makefile ---------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../../..
LIBRARYNAME = LLVMAMDGPUCodeGen
TARGET = AMDGPU

# Make sure that tblgen is run, first thing.
BUILT_SOURCES = AMDGPUGenRegisterInfo.inc AMDGPUGenInstrInfo.inc \
		AMDGPUGenDAGISel.inc  AMDGPUGenSubtargetInfo.inc \
		AMDGPUGenCodeEmitter.inc AMDGPUGenCallingConv.inc \
		AMDGPUGenIntrinsics.inc AMDGPUGenDFAPacketizer.inc \

DIRS = TargetInfo MCTargetDesc

include $(LEVEL)/Makefile.common
