#===-- lib/Target/X86/Makefile.am --------------------------*- Makefile -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
#===------------------------------------------------------------------------===#

include $(top_srcdir)/Makefile.rules.am

libexec_PROGRAMS = LLVMX86.o

BUILT_SOURCES = \
  X86GenRegisterInfo.h.inc \
  X86GenRegisterNames.inc \
  X86GenRegisterInfo.inc \
  X86GenInstrNames.inc \
  X86GenInstrInfo.inc \
  X86GenATTAsmWriter.inc \
  X86GenIntelAsmWriter.inc

LLVMX86_o_SOURCES = \
  X86AsmPrinter.cpp \
  X86CodeEmitter.cpp \
  X86FloatingPoint.cpp \
  X86InstrInfo.cpp \
  X86ISelPattern.cpp \
  X86ISelSimple.cpp \
  X86PeepholeOpt.cpp \
  X86RegisterInfo.cpp \
  X86TargetMachine.cpp

LIBS=

$(BUILT_SOURCES) : $(LLVM_TDFILES) $(TBLGEN)
