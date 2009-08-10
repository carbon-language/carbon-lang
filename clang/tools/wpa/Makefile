LEVEL = ../../../..

TOOLNAME = clang-wpa
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
CXXFLAGS = -fno-rtti

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.config

LINK_COMPONENTS := bitreader mc
USEDLIBS = clangFrontend.a clangSema.a clangAST.a clangLex.a clangBasic.a clangAnalysis.a clangIndex.a

include $(LLVM_SRC_ROOT)/Makefile.rules
