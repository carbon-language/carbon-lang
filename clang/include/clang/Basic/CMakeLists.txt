macro(clang_diag_gen component)
  tablegen(Diagnostic${component}Kinds.inc 
	   -gen-clang-diags-defs -clang-component=${component})
  add_custom_target(ClangDiagnostic${component}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Diagnostic${component}Kinds.inc)
endmacro(clang_diag_gen)

set(LLVM_TARGET_DEFINITIONS Diagnostic.td)
clang_diag_gen(Analysis)
clang_diag_gen(AST)
clang_diag_gen(Common)
clang_diag_gen(Driver)
clang_diag_gen(Frontend)
clang_diag_gen(Lex)
clang_diag_gen(Parse)
clang_diag_gen(Sema)
tablegen(DiagnosticGroups.inc
         -gen-clang-diag-groups)
add_custom_target(ClangDiagnosticGroups
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/DiagnosticGroups.inc)
