CLANG_LEVEL := ../../..
BUILT_SOURCES = StmtNodes.inc DeclNodes.inc
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

INPUT_TDS = $(TD_SRC_DIR)/StmtNodes.td $(TD_SRC_DIR)/DeclNodes.td

$(ObjDir)/StmtNodes.inc.tmp : $(TD_SRC_DIR)/StmtNodes.td $(TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang statement node tables with tblgen"
	$(Verb) $(TableGen) -gen-clang-stmt-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/DeclNodes.inc.tmp : $(TD_SRC_DIR)/DeclNodes.td $(TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang declaration node tables with tblgen"
	$(Verb) $(TableGen) -gen-clang-decl-nodes -o $(call SYSPATH, $@) $<
