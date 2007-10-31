LEVEL = ../../..
CPPFLAGS += -I$(PROJ_SRC_DIR)/../include
CXXFLAGS = -fno-rtti

TOOLNAME = clang
USEDLIBS = clangCodeGen.a clangAnalysis.a clangRewrite.a clangSEMA.a \
           clangAST.a clangParse.a clangLex.a clangBasic.a LLVMX86 \
           LLVMCore.a LLVMSupport.a LLVMSystem.a \
           LLVMBitWriter.a LLVMBitReader.a LLVMTarget.a LLVMCodeGen.a \
           LLVMSelectionDAG.a LLVMScalarOpts.a LLVMTransformUtils.a \
           LLVMCore.a LLVMAnalysis.a
	
	

include $(LEVEL)/Makefile.common
