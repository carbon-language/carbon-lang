set(LLVM_USED_LIBS clangBasic clangAST clangParse clangFrontend clangRewrite)

add_clang_library(clangARCMigrate
  ARCMT.cpp
  ARCMTActions.cpp
  FileRemapper.cpp
  TransAllocCopyWithZone.cpp
  TransARCAssign.cpp
  TransAutoreleasePool.cpp
  TransBlockObjCVariable.cpp
  TransDeallocMethod.cpp
  TransEmptyStatements.cpp
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
