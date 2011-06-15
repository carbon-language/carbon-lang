set(LLVM_USED_LIBS clangBasic clangAST clangParse clangFrontend clangRewrite)

add_clang_library(clangARCMigrate
  ARCMT.cpp
  FileRemapper.cpp
  TransformActions.cpp
  Transforms.cpp
  )

add_dependencies(clangARCMigrate
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNodes)
