CLANG_LEVEL := ../../../..
TOOLNAME = clang-rename
include $(CLANG_LEVEL)/../../Makefile.config
# Include this here so we can get the configuration of the targets that have
# been configured for construction. We have to do this early so we can set up
# LINK_COMPONENTS before including Makefile.rules
include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader bitwriter \
                   instrumentation ipo support mc objcarcopts option
USEDLIBS = clangRename.a clangFrontend.a clangSerialization.a clangDriver.a \
           clangTooling.a clangToolingCore.a \
	   clangParse.a clangSema.a clangIndex.a \
           clangStaticAnalyzerFrontend.a clangStaticAnalyzerCheckers.a \
           clangStaticAnalyzerCore.a clangAnalysis.a clangRewriteFrontend.a \
           clangRewrite.a clangEdit.a clangAST.a clangLex.a clangBasic.a \
           clangCodeGen.a

include $(CLANG_LEVEL)/Makefile

