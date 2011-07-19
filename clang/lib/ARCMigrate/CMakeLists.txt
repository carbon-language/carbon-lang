set(LLVM_USED_LIBS clangBasic clangAST clangParse clangFrontend clangRewrite)

add_clang_library(clangARCMigrate
  ARCMT.cpp
  ARCMTActions.cpp
  FileRemapper.cpp
  PlistReporter.cpp
  TransAPIUses.cpp
  TransARCAssign.cpp
  TransAutoreleasePool.cpp
  TransBlockObjCVariable.cpp
  TransEmptyStatementsAndDealloc.cpp
  TransformActions.cpp
  Transforms.cpp
  TransProperties.cpp
  TransRetainReleaseDealloc.cpp
  TransUnbridgedCasts.cpp
  TransUnusedInitDelegate.cpp
  TransZeroOutPropsInDealloc.cpp
  )

add_dependencies(clangARCMigrate
  ClangAttrClasses
  ClangAttrList
  ClangDeclNodes
  ClangStmtNodes)
