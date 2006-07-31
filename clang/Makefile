LEVEL = ../..
PARALLEL_DIRS := Basic Lex Parse
CPPFLAGS += -I$(LEVEL)/tools/clang/include

CXXFLAGS = -fno-rtti -fno-exceptions

TOOLNAME = clang

USEDLIBS = clangParse.a clangLex.a clangBasic.a LLVMSupport.a LLVMSystem.a

include $(LEVEL)/Makefile.common
