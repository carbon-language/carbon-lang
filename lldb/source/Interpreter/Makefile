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
-include $(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.d

# edit-swig-python-wrapper-file.py needs $(SRCROOT)
export SRCROOT := $(PROJ_SRC_DIR)/$(LLDB_LEVEL)
TARGET_DIR := $(PROJ_OBJ_DIR)

PYTHON_DIR := $(LibDir)/python/lldb
ifeq ($(HOST_OS),Darwin)
PYTHON_DEST_DIR := $(DESTDIR)/$(call realpath,/Library/Python/$(shell python -c 'import sys; print sys.version[:3]')/site-packages)/lldb
else
PYTHON_DEST_DIR := $(DESTDIR)/$(call realpath,$(shell python -c 'import sys; print sys.exec_prefix')/lib/python$(shell python -c 'import sys; print sys.version[:3]')/site-packages)/lldb
endif
LLDB_SWIG_INCLUDE_DIRS:= -I"$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/include" -I./.
LIBLLDB := $(LibDir)/liblldb$(SHLIBEXT)
INST_LIBLLDB := $(PROJ_libdir)/liblldb$(SHLIBEXT)

# Subpackages of the main LLDB package
LLDB_SUBPACKAGES := FORMATTERS FORMATTERS_CPP FORMATTERS_OBJC RUNTIME RUNTIME_OBJC UTILS
# lldb.formatters
LLDB_PACKAGE_FORMATTERS := formatters
LLDB_PACKAGE_FORMATTERS_FILES := $(addprefix \
  $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/summaries/cocoa/, \
  cache.py metrics.py attrib_fromdict.py Logger.py)
# lldb.formatters.cpp
LLDB_PACKAGE_FORMATTERS_CPP := formatters/cpp
LLDB_PACKAGE_FORMATTERS_CPP_FILES := $(addprefix \
  $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/synthetic/,gnu_libstdcpp.py libcxx.py)
# lldb.formatters.objc
LLDB_PACKAGE_FORMATTERS_OBJC := formatters/objc
LLDB_PACKAGE_FORMATTERS_OBJC_FILES := \
  $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/summaries/cocoa/Selector.py \
  $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/summaries/objc.py \
  $(addprefix $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/summaries/cocoa/, \
    Class.py CFArray.py CFBag.py CFBinaryHeap.py CFBitVector.py \
    CFDictionary.py CFString.py NSBundle.py NSData.py NSDate.py \
    NSException.py NSIndexSet.py NSMachPort.py NSNotification.py \
    NSNumber.py NSSet.py NSURL.py \
  )
# lldb.runtime
LLDB_PACKAGE_RUNTIME := runtime
LLDB_PACKAGE_RUNTIME_FILES :=
# lldb.runtime.objc
LLDB_PACKAGE_RUNTIME_OBJC := runtime/objc
LLDB_PACKAGE_RUNTIME_OBJC_FILES := \
  $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/summaries/cocoa/objc_runtime.py
# lldb.utils
LLDB_PACKAGE_UTILS := utils
LLDB_PACKAGE_UTILS_FILES := \
  $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/python/symbolication.py

ifeq ($(HOST_OS),Darwin)
# Additional packages for Darwin/Mac OS X
LLDB_SUBPACKAGES += MACOSX
# lldb.macosx
LLDB_PACKAGE_MACOSX := macosx
LLDB_PACKAGE_MACOSX_FILES := $(addprefix \
  $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/, python/crashlog.py darwin/heap_find/heap.py \
)
endif

LLDB_ALL_SUBPACKAGE_FILES := $(foreach subpackage,$(LLDB_SUBPACKAGES),$(LLDB_PACKAGE_$(subpackage)_FILES))
LLDB_ALL_INSTALLED_SUBPACKAGE_FILES := $(foreach subpackage,$(LLDB_SUBPACKAGES),$(addprefix $(LLDB_PACKAGE_$(subpackage))/,$(notdir $(LLDB_PACKAGE_$(subpackage)_FILES))))

comma := ,
LLDB_COPY_PACKAGE = $(Verb) \
$(foreach subpackage,$(LLDB_SUBPACKAGES), \
  init_file="$(PYTHON_DIR)/$(LLDB_PACKAGE_$(subpackage))/__init__.py"; \
  $(MKDIR) "$(PYTHON_DIR)/$(LLDB_PACKAGE_$(subpackage))"; \
  $(foreach file,$(LLDB_PACKAGE_$(subpackage)_FILES), \
    $(CP) "$(file)" "$(PYTHON_DIR)/$(LLDB_PACKAGE_$(subpackage))"; \
  ) \
  echo "__all__ = [$(patsubst %,\"%\"$(comma),\
    $(basename $(notdir $(LLDB_PACKAGE_$(subpackage)_FILES))))]" >$$init_file; \
  echo "for x in __all__:" >>$$init_file; \
  echo "    __import__('lldb.$(subst /,.,$(LLDB_PACKAGE_$(subpackage))).'+x)" >>$$init_file; \
)

LLDB_INSTALL_SUBPACKAGES = $(Verb) \
$(foreach subpackage,$(LLDB_SUBPACKAGES), \
  $(MKDIR) $(PYTHON_DEST_DIR)/$(LLDB_PACKAGE_$(subpackage)); \
  $(DataInstall) $(PYTHON_DIR)/$(LLDB_PACKAGE_$(subpackage))/__init__.py \
                 $(PYTHON_DEST_DIR)/$(LLDB_PACKAGE_$(subpackage))/__init__.py; \
) \
$(foreach file,$(LLDB_ALL_INSTALLED_SUBPACKAGE_FILES), \
  $(DataInstall) $(PYTHON_DIR)/$(file) $(PYTHON_DEST_DIR)/$(file); \
)

ifeq ($(HOST_OS),Darwin)
# Install the heap_find sources, too.
LLDB_INSTALL_SUBPACKAGES += $(MKDIR) $(PYTHON_DEST_DIR)/macosx/heap/; \
$(DataInstall) $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/darwin/heap_find/heap/heap_find.cpp \
               $(PYTHON_DEST_DIR)/macosx/heap/heap_find.cpp; \
$(DataInstall) $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/examples/darwin/heap_find/heap/Makefile \
			   $(PYTHON_DEST_DIR)/macosx/heap/Makefile;
endif

# We need Swig to process stdint.h, but by default it will not inspect system
# include directories.  The following should cover the standard locations on
# most platforms.
LLDB_SWIG_INCLUDE_DIRS += -I"/usr/local/include"
LLDB_SWIG_INCLUDE_DIRS += -I"/usr/include"

# On Darwin, stdint.h might only be in the compiler subdirectory. Unfortunately,
# swig doesn't currently seem able to understand Clang's stdint.h, so we have to
# point at the GCC one.
ifeq ($(HOST_OS),Darwin)
LLDB_SWIG_INCLUDE_DIRS += -I"/usr/include/gcc/darwin/4.2"
endif

ifndef DISABLE_AUTO_DEPENDENCIES

SWIG_DEPEND_OPTIONS = -MMD -MF "$(PROJ_OBJ_DIR)/$*.cpp.d.tmp"
CPP_DEPEND_MOVEFILE = then $(MV) -f "$(PROJ_OBJ_DIR)/$*.cpp.d.tmp" "$(PROJ_OBJ_DIR)/$*.cpp.d"; \
                      else $(RM) -f "$(PROJ_OBJ_DIR)/$*.cpp.d.tmp"; exit 1; fi

LLDBWrapPython.cpp: $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/modify-python-lldb.py \
                    $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/edit-swig-python-wrapper-file.py \
                    $(wildcard $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/interface/*.i)
	$(Echo) Generating LLDBWrapPython.cpp
	$(Verb) $(MKDIR) $(PYTHON_DIR)
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/build-swig-wrapper-classes.sh" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "$(TARGET_DIR)" "$(PYTHON_DIR)" "$(PYTHON_DIR)"
	$(Verb) "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/finish-swig-wrapper-classes.sh" "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)" "$(TARGET_DIR)" "$(PYTHON_DIR)" "$(PYTHON_DIR)"
#	$(Verb) if swig -c++ -shadow -python $(LLDB_SWIG_INCLUDE_DIRS) \
#          -D__STDC_LIMIT_MACROS -outdir "$(PYTHON_DIR)"   \
#          $(SWIG_DEPEND_OPTIONS) \
#          -o LLDBWrapPython.cpp "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/lldb.swig" ; \
#          $(CPP_DEPEND_MOVEFILE)
#	$(Verb) python "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/modify-python-lldb.py" \
#		  "$(PYTHON_DIR)"
#	$(Verb) python "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/edit-swig-python-wrapper-file.py" \
#		  "$(PROJ_OBJ_DIR)"
#	$(Verb) if test -f "$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.edited"; then \
#		  $(MV) "$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.edited" \
#				"$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp"; fi
#	$(Verb) $(MV) "$(PYTHON_DIR)/lldb.py" "$(PYTHON_DIR)/__init__.py"
#	$(Verb) $(CP) "$(PROJ_SRC_DIR)/embedded_interpreter.py" "$(PYTHON_DIR)"
#	$(Verb) $(RM) -f $(PYTHON_DIR)/_lldb.so
#	$(Verb) $(AliasTool) $(LIBLLDB) $(PYTHON_DIR)/_lldb.so
#	$(LLDB_COPY_PACKAGE)

else

LLDBWrapPython.cpp: $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/modify-python-lldb.py \
                    $(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/edit-swig-python-wrapper-file.py
	$(Echo) Generating LLDBWrapPython.cpp
	$(Verb) $(MKDIR) $(PYTHON_DIR)
	$(Verb) swig -c++ -shadow -python $(LLDB_SWIG_INCLUDE_DIRS) \
          -D__STDC_LIMIT_MACROS -outdir "$(PYTHON_DIR)"   \
          -o LLDBWrapPython.cpp "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/lldb.swig"
	$(Verb) python "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/modify-python-lldb.py" \
		  "$(PYTHON_DIR)"
	$(Verb) python "$(PROJ_SRC_DIR)/$(LLDB_LEVEL)/scripts/Python/edit-swig-python-wrapper-file.py" \
		  "$(PROJ_OBJ_DIR)"
	$(Verb) if test -f "$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.edited"; then \
		  $(MV) "$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp.edited" \
				"$(PROJ_OBJ_DIR)/LLDBWrapPython.cpp"; fi
	$(Verb) $(MV) "$(PYTHON_DIR)/lldb.py" "$(PYTHON_DIR)/__init__.py"
	$(Verb) $(CP) "$(PROJ_SRC_DIR)/embedded_interpreter.py" "$(PYTHON_DIR)"
	$(Verb) $(RM) -f $(PYTHON_DIR)/_lldb.so
	$(Verb) $(AliasTool) $(LIBLLDB) $(PYTHON_DIR)/_lldb.so
	$(LLDB_COPY_PACKAGE)

endif

install-local:: $(PYTHON_DIR)/__init__.py $(PYTHON_DIR)/embedded_interpreter.py \
                $(addprefix $(PYTHON_DIR)/,$(LLDB_ALL_INSTALLED_SUBPACKAGE_FILES))
	$(Echo) Installing $(BuildMode) LLDB python modules
	$(Verb) $(MKDIR) $(PYTHON_DEST_DIR)
	$(Verb) $(DataInstall) $(PYTHON_DIR)/__init__.py $(PYTHON_DEST_DIR)/__init__.py
	$(Verb) $(DataInstall) $(PYTHON_DIR)/embedded_interpreter.py $(PYTHON_DEST_DIR)/embedded_interpreter.py
	$(Verb) $(RM) -f $(PYTHON_DEST_DIR)/_lldb.so
	$(Verb) $(AliasTool) $(INST_LIBLLDB) $(PYTHON_DEST_DIR)/_lldb.so
	$(LLDB_INSTALL_SUBPACKAGES)

clean-local::
	$(Verb) $(RM) -f LLDBWrapPython.cpp $(PYTHON_DIR)/_lldb.so \
	          $(PYTHON_DIR)/embedded_interpreter.py $(PYTHON_DIR)/__init__.py \
			  $(addprefix $(PYTHON_DIR)/,$(LLDB_ALL_INSTALLED_SUBPACKAGE_FILES))
