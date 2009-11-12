LEVEL = ../../../../..
BUILT_SOURCES = DiagnosticAnalysisKinds.inc DiagnosticASTKinds.inc \
	DiagnosticCommonKinds.inc DiagnosticDriverKinds.inc \
	DiagnosticFrontendKinds.inc DiagnosticLexKinds.inc \
	DiagnosticParseKinds.inc DiagnosticSemaKinds.inc \
        DiagnosticGroups.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(LEVEL)/Makefile.common

INPUT_TDS = $(wildcard $(PROJ_SRC_DIR)/Diagnostic*.td)

$(ObjDir)/Diagnostic%Kinds.inc.tmp : Diagnostic.td Diagnostic%Kinds.td $(TBLGEN)
	$(Echo) "Building Clang $(patsubst Diagnostic%Kinds.inc.tmp,%,$(@F)) diagnostic tables with tblgen"
	$(Verb) -$(MKDIR) $(@D)
	$(Verb) $(TableGen) -gen-clang-diags-defs -clang-component=$(patsubst Diagnostic%Kinds.inc.tmp,%,$(@F)) -o $(call SYSPATH, $@) $<

$(ObjDir)/DiagnosticGroups.inc.tmp : Diagnostic.td DiagnosticGroups.td $(INPUT_TDS) $(TBLGEN)
	$(Echo) "Building Clang diagnostic groups with tblgen"
	$(Verb) -$(MKDIR) $(@D)
	$(Verb) $(TableGen) -gen-clang-diag-groups -o $(call SYSPATH, $@) $<


