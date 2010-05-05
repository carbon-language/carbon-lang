LEVEL = ../../../../..
BUILT_SOURCES = StmtNodes.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(LEVEL)/Makefile.common

INPUT_TDS = $(PROJ_SRC_DIR)/StmtNodes.td

$(ObjDir)/StmtNodes.inc.tmp : StmtNodes.td $(TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang statement node tables with tblgen"
	$(Verb) $(TableGen) -gen-clang-stmt-nodes -o $(call SYSPATH, $@) $<

