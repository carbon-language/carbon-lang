set(LLVM_USED_LIBS clangBasic clangAST clangLex clangSema)

add_clang_library(clangParse
  ParseAST.cpp
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

add_dependencies(clangParse ClangAttrClasses ClangAttrList ClangDeclNodes ClangDiagnosticParse ClangStmtNodes)
