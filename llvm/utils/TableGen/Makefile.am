##===- utils/TableGen/Makefile.am --------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file was developed by Reid Spencer and is distributed under the 
# University of Illinois Open Source License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

include $(top_srcdir)/Makefile_config

bin_PROGRAMS = TableGen

BUILT_SOURCES  = FileLexer.cpp FileParser.cpp FileParser.h

TableGen_SOURCES =AsmWriterEmitter.cpp \
                  CodeEmitterGen.cpp \
                  CodeEmitterGen.h \
                  CodeGenInstruction.h \
                  CodeGenRegisters.h \
                  CodeGenTarget.cpp \
                  CodeGenTarget.h \
                  InstrInfoEmitter.cpp \
                  InstrInfoEmitter.h \
                  InstrSelectorEmitter.cpp \
                  InstrSelectorEmitter.h \
                  Record.cpp \
                  Record.h \
                  RegisterInfoEmitter.cpp \
                  RegisterInfoEmitter.h \
                  TableGenBackend.cpp \
                  TableGenBackend.h \
                  TableGen.cpp \
                  $(BUILT_SOURCES)

TableGen_LDADD =  $(call GETLIBS,Support,System) 
