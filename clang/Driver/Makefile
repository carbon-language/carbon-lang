LEVEL = ../../..
CPPFLAGS += -I$(PROJ_SRC_DIR)/../include
CXXFLAGS = -fno-rtti

TOOLNAME = clang
USEDLIBS = clangCodeGen.a clangAnalysis.a clangRewrite.a clangSEMA.a \
           clangAST.a clangParse.a clangLex.a clangBasic.a \
           LLVMCore.a LLVMSupport.a LLVMSystem.a \
           LLVMBitWriter.a LLVMBitReader.a LLVMCodeGen.a LLVMAnalysis.a \
           LLVMTarget.a

include $(LEVEL)/Makefile.common
