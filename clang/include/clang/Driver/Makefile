CLANG_LEVEL := ../../..
BUILT_SOURCES = Options.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/Options.inc.tmp : Options.td CC1Options.td CLCompatOptions.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang Driver Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<
