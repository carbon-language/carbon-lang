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
ifeq ($(HOST_OS),Darwin)
PYTHON_DEST_DIR := /Library/Python/$(shell python -c 'import sys; print sys.version[:3]')/site-packages
else
PYTHON_DEST_DIR := $(shell python -c 'import sys; print sys.exec_prefix')/lib/python$(shell python -c 'import sys; print sys.version[:3]')/site-packages
endif
LLDB_SWIG_INCLUDE_DIRS:= -I"$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/include" -I./.
LIBLLDB := $(DESTDIR)$(PROJ_libdir)/liblldb$(SHLIBEXT)

# We need Swig to process stdint.h, but by default it will not inspect system
# include directories.  The following should cover the standard locations on
# most platforms.
LLDB_SWIG_INCLUDE_DIRS += -I"/usr/local/include"
LLDB_SWIG_INCLUDE_DIRS += -I"/usr/include"

LLDBWrapPython.cpp:
	$(Echo) Generating LLDBWrapPython.cpp
	$(Verb) swig -c++ -shadow -python $(LLDB_SWIG_INCLUDE_DIRS) \
          -D__STDC_LIMIT_MACROS -outdir "$(LLDB_BIN_DIR)"   \
          -o LLDBWrapPython.cpp "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/lldb.swig"
	$(Verb) python "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/modify-python-lldb.py" \
		  "$(LLDB_BIN_DIR)"
	$(Verb) python "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/edit-swig-python-wrapper-file.py" \
		  "$(PROJ_OBJ_DIR)"
	$(Verb) if test -f "$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.edited"; then \
		  mv "$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.edited" \
			 "$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp"; fi
	$(Verb) cp "$(PROJ_SRC_DIR)/embedded_interpreter.py" "$(PYTHON_DIR)"

install-local:: $(PYTHON_DIR)/lldb.py $(PYTHON_DIR)/embedded_interpreter.py $(LIBLLDB)
	$(Echo) Installing $(BuildMode) LLDB python modules
	$(Verb) $(MKDIR) $(PYTHON_DEST_DIR)/lib-dynload
	$(Verb) $(DataInstall) $(PYTHON_DIR)/lldb.py $(PYTHON_DEST_DIR)/lldb.py
	$(Verb) $(DataInstall) $(PYTHON_DIR)/embedded_interpreter.py $(PYTHON_DEST_DIR)/embedded_interpreter.py
	$(Verb) $(RM) -f $(PYTHON_DEST_DIR)/lib-dynload/_lldb.so
	$(Verb) $(AliasTool) $(LIBLLDB) $(PYTHON_DEST_DIR)/lib-dynload/_lldb.so
