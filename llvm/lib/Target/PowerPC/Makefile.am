#===-- lib/Target/PowerPC/Makefile.am ----------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile_config

lib_LIBRARIES = libLLVMPowerPC.a

BUILT_SOURCES = \
  PowerPCGenInstrNames.inc \
  PowerPCGenRegisterNames.inc \
  PowerPCGenAsmWriter.inc \
  PPC32GenCodeEmitter.inc \
  PPC32GenRegisterInfo.h.inc \
  PPC32GenRegisterInfo.inc \
  PPC32GenInstrInfo.inc \
  PPC64GenRegisterInfo.h.inc \
  PPC64GenRegisterInfo.inc \
  PPC64GenInstrInfo.inc

libLLVMPowerPC_a_SOURCES = \
  PowerPCAsmPrinter.cpp \
  PowerPCBranchSelector.cpp \
  PowerPCTargetMachine.cpp \
  PPC32CodeEmitter.cpp \
  PPC32InstrInfo.cpp \
  PPC32ISelSimple.cpp \
  PPC32RegisterInfo.cpp \
  PPC64CodeEmitter.cpp \
  PPC64InstrInfo.cpp \
  PPC64ISelSimple.cpp \
  PPC64RegisterInfo.cpp

PRELINK=libLLVMPowerPC.a

$(BUILT_SOURCES) : $(LLVM_TDFILES) $(TBLGEN)
