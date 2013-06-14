CLANG_LEVEL := ../../..
BUILT_SOURCES = Options.inc CC1AsOptions.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/Options.inc.tmp : Options.td CC1Options.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang Driver Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<

$(ObjDir)/CC1AsOptions.inc.tmp : CC1AsOptions.td $(LLVM_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang CC1 Assembler Option tables with tblgen"
	$(Verb) $(LLVMTableGen) -gen-opt-parser-defs -o $(call SYSPATH, $@) $<
