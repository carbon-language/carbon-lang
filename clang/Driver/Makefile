LEVEL = ../../..
CPPFLAGS += -I$(PROJ_SRC_DIR)/../include
CXXFLAGS = -fno-rtti

TOOLNAME = clang
USEDLIBS = clangCodeGen.a clangAnalysis.a clangRewrite.a clangSEMA.a \
           clangAST.a clangParse.a clangLex.a clangBasic.a \
           LLVMAnalysis.a \
           LLVMCore.a LLVMSupport.a LLVMSystem.a \
           LLVMBitWriter.a LLVMBitReader.a LLVMCodeGen.a LLVMTarget.a 
	
	

include $(LEVEL)/Makefile.common
