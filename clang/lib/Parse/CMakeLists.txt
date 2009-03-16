set(LLVM_NO_RTTI 1)

add_clang_library(clangParse
  AttributeList.cpp
  DeclSpec.cpp
  MinimalAction.cpp
  ParseCXXInlineMethods.cpp
  ParseDecl.cpp
  ParseDeclCXX.cpp
  ParseExpr.cpp
  ParseExprCXX.cpp
  ParseInit.cpp
  ParseObjc.cpp
  ParsePragma.cpp
  Parser.cpp
  ParseStmt.cpp
  ParseTentative.cpp
  ParseTemplate.cpp
  )

add_dependencies(clangParse ClangDiagnosticParse)
