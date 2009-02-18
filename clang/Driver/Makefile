LEVEL = ../../..
TOOLNAME = clang
CPPFLAGS += -I$(PROJ_SRC_DIR)/../include
CXXFLAGS = -fno-rtti

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

# Optimize startup time of the app by not exporting all of the weak symbols
# from the binary.  This reduces dyld startup time by 4x.
ifeq ($(OS),Darwin)
LD.Flags += -Wl,-exported_symbol -Wl,_main
endif

LINK_COMPONENTS := $(TARGETS_TO_BUILD) bitreader bitwriter codegen ipo selectiondag
USEDLIBS = clangCodeGen.a clangAnalysis.a clangRewrite.a clangSema.a	\
           clangDriver.a clangAST.a clangParse.a clangLex.a		\
           clangBasic.a 

include $(LLVM_SRC_ROOT)/Makefile.rules
