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
  ParseStmt.cpp
  ParseTemplate.cpp
  ParseTentative.cpp
  Parser.cpp
  )

add_dependencies(clangParse ClangDiagnosticParse)
