##===- tools/extra/loop-convert/Makefile ----sssss----------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../..

TOOLNAME = cpp11-migrate
NO_INSTALL = 1

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(CLANG_LEVEL)/../../Makefile.config

SOURCES = Cpp11Migrate.cpp Transforms.cpp

# For each Transform subdirectory add to SOURCES and BUILT_SOURCES.
# BUILT_SOURCES ensures a subdirectory is created to house object files from
# transform subdirectories. See below for more on .objdir.
SOURCES += $(addprefix LoopConvert/,$(notdir $(wildcard $(PROJ_SRC_DIR)/LoopConvert/*.cpp)))
BUILT_SOURCES = $(ObjDir)/LoopConvert/.objdir

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser support mc
USEDLIBS = clangTooling.a clangFrontend.a clangSerialization.a clangDriver.a \
		   clangRewriteFrontend.a clangRewriteCore.a clangParse.a \
		   clangSema.a clangAnalysis.a \
		   clangAST.a clangASTMatchers.a clangEdit.a clangLex.a clangBasic.a

include $(CLANG_LEVEL)/Makefile

# BUILT_SOURCES gets used as a prereq for many top-level targets. However, at
# the point those targets are defined, $(ObjDir) hasn't been defined and so the
# directory to create becomes /<name>/ which is not what we want. So instead,
# this .objdir recipe is defined at at point where $(ObjDir) is defined and
# it's specialized to $(ObjDir) to ensure it only works on targets we want it
# to.
$(ObjDir)/%.objdir:
	$(Verb) $(MKDIR) $(ObjDir)/$* > /dev/null
	$(Verb) $(DOTDIR_TIMESTAMP_COMMAND) > $@

