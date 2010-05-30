LEVEL = ../../../../..
BUILT_SOURCES = StmtNodes.inc DeclNodes.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(LEVEL)/Makefile.common

INPUT_TDS = $(PROJ_SRC_DIR)/StmtNodes.td $(PROJ_SRC_DIR)/DeclNodes.td

$(ObjDir)/StmtNodes.inc.tmp : StmtNodes.td $(TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang statement node tables with tblgen"
	$(Verb) $(TableGen) -gen-clang-stmt-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/DeclNodes.inc.tmp : DeclNodes.td $(TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang declaration node tables with tblgen"
	$(Verb) $(TableGen) -gen-clang-decl-nodes -o $(call SYSPATH, $@) $<
