LEVEL = ../../../..

TOOLNAME = clang-wpa
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_OBJ_DIR)/../../include
CXXFLAGS = -fno-rtti
NO_INSTALL = 1

# No plugins, optimize startup time.
TOOL_NO_EXPORTS = 1

include $(LEVEL)/Makefile.config

LINK_COMPONENTS := bitreader mc core
USEDLIBS = clangIndex.a clangFrontend.a clangDriver.a clangSema.a \
	   clangAnalysis.a clangAST.a clangParse.a clangLex.a clangBasic.a

include $(LLVM_SRC_ROOT)/Makefile.rules
