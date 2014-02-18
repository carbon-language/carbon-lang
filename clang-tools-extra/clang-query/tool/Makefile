##===- tools/extra/clang-query/tool/Makefile ---------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../../..
include $(CLANG_LEVEL)/../../Makefile.config

TOOLNAME = clang-query

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

SOURCES = ClangQuery.cpp

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc mcparser option
USEDLIBS = clangQuery.a clangDynamicASTMatchers.a clangFormat.a clangTooling.a \
	   clangFrontend.a clangSerialization.a clangDriver.a clangRewriteFrontend.a \
	   LLVMLineEditor.a clangRewriteCore.a clangParse.a clangSema.a clangAnalysis.a \
	   clangAST.a clangASTMatchers.a clangEdit.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

CPP.Flags += -I$(PROJ_SRC_DIR)/..

# BUILT_SOURCES gets used as a prereq for many top-level targets. However, at
# the point those targets are defined, $(ObjDir) hasn't been defined and so the
# directory to create becomes /<name>/ which is not what we want. So instead,
# this .objdir recipe is defined at at point where $(ObjDir) is defined and
# it's specialized to $(ObjDir) to ensure it only works on targets we want it
# to.
$(ObjDir)/%.objdir:
	$(Verb) $(MKDIR) $(ObjDir)/$* > /dev/null
	$(Verb) $(DOTDIR_TIMESTAMP_COMMAND) > $@

