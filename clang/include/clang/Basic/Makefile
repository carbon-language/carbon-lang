LEVEL = ../../../../..
BUILT_SOURCES = DiagnosticAnalysisKinds.inc DiagnosticASTKinds.inc \
	DiagnosticCommonKinds.inc DiagnosticDriverKinds.inc \
	DiagnosticFrontendKinds.inc DiagnosticLexKinds.inc \
	DiagnosticParseKinds.inc DiagnosticSemaKinds.inc \
        DiagnosticGroups.inc

TABLEGEN_INC_FILES_COMMON = 1

include $(LEVEL)/Makefile.common

$(ObjDir)/Diagnostic%Kinds.inc.tmp : Diagnostic.td Diagnostic%Kinds.td $(TBLGEN)
	$(Echo) "Building Clang $(patsubst Diagnostic%Kinds.inc.tmp,%,$(@F)) diagnostic tables with tblgen"
	$(Verb) -$(MKDIR) $(@D)
	$(Verb) $(TableGen) -gen-clang-diags-defs -clang-component=$(patsubst Diagnostic%Kinds.inc.tmp,%,$(@F)) -o $(call SYSPATH, $@) $<

$(ObjDir)/DiagnosticGroups.inc.tmp : Diagnostic.td DiagnosticGroups.td $(wildcard Diagnostic*.td) $(TBLGEN)
	$(Echo) "Building Clang diagnostic groups with tblgen"
	$(Verb) -$(MKDIR) $(@D)
	$(Verb) $(TableGen) -gen-clang-diag-groups -o $(call SYSPATH, $@) $<


