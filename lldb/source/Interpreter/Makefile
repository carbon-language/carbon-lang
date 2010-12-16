##===- source/Interpreter/Makefile  ------------------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../..
LIBRARYNAME := lldbInterpreter
BUILD_ARCHIVE = 1

BUILT_SOURCES := LLDBWrapPython.cpp

include $(LLDB_LEVEL)/Makefile

LLDB_PYTHON_SWIG_CPP = $(PROJ_OBJ_ROOT)/$(BuildMode)/LLDBWrapPython.cpp
LLDB_BIN_DIR := $(PROJ_OBJ_ROOT)/$(BuildMode)/bin
PYTHON_DIR := $(LLDB_BIN_DIR)

LLDBWrapPython.cpp:
	swig -c++ -shadow -python -I"$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/include"            \
          -I./. -I"/usr/local/include" -D__STDC_LIMIT_MACROS -outdir "$(LLDB_BIN_DIR)" \
          -o LLDBWrapPython.cpp "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/lldb.swig"
	cp "$(PROJ_SRC_DIR)/embedded_interpreter.py" "$(PYTHON_DIR)"
