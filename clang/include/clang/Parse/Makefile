CLANG_LEVEL := ../../..
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic
BUILT_SOURCES = AttrParserStringSwitches.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/AttrParserStringSwitches.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                                   $(ObjDir)/.dir
	$(Echo) "Building Clang parser-related attribute string switches"
	$(Verb) $(ClangTableGen) -gen-clang-attr-parser-string-switches -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<
