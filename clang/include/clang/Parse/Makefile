CLANG_LEVEL := ../../..
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic
BUILT_SOURCES = AttrIdentifierArg.inc AttrLateParsed.inc AttrTypeArg.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/AttrIdentifierArg.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                                   $(ObjDir)/.dir
	$(Echo) "Building Clang attribute identifier argument table with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-identifier-arg-list -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/AttrTypeArg.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                                   $(ObjDir)/.dir
	$(Echo) "Building Clang attribute type argument table with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-type-arg-list -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/AttrLateParsed.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                                   $(ObjDir)/.dir
	$(Echo) "Building Clang attribute late-parsed table with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-late-parsed-list -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<
