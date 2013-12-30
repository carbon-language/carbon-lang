CLANG_LEVEL := ../../..
TD_SRC_DIR = $(PROJ_SRC_DIR)/../Basic
BUILT_SOURCES = Attrs.inc AttrImpl.inc AttrDump.inc AttrVisitor.inc \
                StmtNodes.inc DeclNodes.inc \
                CommentNodes.inc CommentHTMLTags.inc \
                CommentHTMLTagsProperties.inc \
                CommentHTMLNamedCharacterReferences.inc \
                CommentCommandInfo.inc \
                CommentCommandList.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(CLANG_LEVEL)/Makefile

$(ObjDir)/Attrs.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang attribute classes with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-classes -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/AttrImpl.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang attribute implementations with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-impl -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/AttrDump.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang attribute dumper with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-dump -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/AttrVisitor.inc.tmp : $(TD_SRC_DIR)/Attr.td $(CLANG_TBLGEN) \
                                $(ObjDir)/.dir
	$(Echo) "Building Clang attribute AST visitor with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-attr-ast-visitor -o $(call SYSPATH, $@) \
		-I $(PROJ_SRC_DIR)/../../ $<

$(ObjDir)/StmtNodes.inc.tmp : $(TD_SRC_DIR)/StmtNodes.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang statement node tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-stmt-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/DeclNodes.inc.tmp : $(TD_SRC_DIR)/DeclNodes.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang declaration node tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-decl-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentNodes.inc.tmp : $(TD_SRC_DIR)/CommentNodes.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang comment node tables with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-nodes -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentHTMLTags.inc.tmp : $(PROJ_SRC_DIR)/CommentHTMLTags.td $(CLANG_TBLGEN) \
                              $(ObjDir)/.dir
	$(Echo) "Building Clang comment HTML tag matchers with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-html-tags -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentHTMLTagsProperties.inc.tmp : $(PROJ_SRC_DIR)/CommentHTMLTags.td \
                                              $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang comment HTML tag properties with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-html-tags-properties -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentHTMLNamedCharacterReferences.inc.tmp : \
                    $(PROJ_SRC_DIR)/CommentHTMLNamedCharacterReferences.td \
                    $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang named character reference translation function with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-html-named-character-references -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentCommandInfo.inc.tmp : $(PROJ_SRC_DIR)/CommentCommands.td \
                                              $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang comment command info with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-command-info -o $(call SYSPATH, $@) $<

$(ObjDir)/CommentCommandList.inc.tmp : $(PROJ_SRC_DIR)/CommentCommands.td \
                                              $(CLANG_TBLGEN) $(ObjDir)/.dir
	$(Echo) "Building Clang list of comment commands with tblgen"
	$(Verb) $(ClangTableGen) -gen-clang-comment-command-list -o $(call SYSPATH, $@) $<

