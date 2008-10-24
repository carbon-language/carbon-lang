LEVEL = ../../..
TOOLNAME = clang
CPPFLAGS += -I$(PROJ_SRC_DIR)/../include
CXXFLAGS = -fno-rtti

# Include this here so we can get the configuration of the targets
# that have been configured for construction. We have to do this 
# early so we can set up LINK_COMPONENTS before including Makefile.rules
include $(LEVEL)/Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD)
USEDLIBS = clangCodeGen.a clangAnalysis.a clangRewrite.a clangSema.a	\
           clangDriver.a clangAST.a clangParse.a clangLex.a		\
           clangBasic.a LLVMCore.a LLVMSupport.a LLVMSystem.a		\
           LLVMBitWriter.a LLVMBitReader.a LLVMSelectionDAG.a		\
           LLVMCodeGen.a LLVMTransformUtils.a LLVMScalarOpts.a		\
           LLVMAnalysis.a LLVMTarget.a LLVMipo.a

include $(LLVM_SRC_ROOT)/Makefile.rules
