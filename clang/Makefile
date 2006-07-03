LEVEL = ../..
PARALLEL_DIRS := Basic Lex
CPPFLAGS += -I$(LEVEL)/tools/clang/include

CXXFLAGS = -fno-rtti -fno-exceptions

TOOLNAME = clang

USEDLIBS = clangLex.a clangBasic.a LLVMSupport.a LLVMSystem.a

include $(LEVEL)/Makefile.common
