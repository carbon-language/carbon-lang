set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  ArgumentCommentCheck.cpp
  AssertSideEffectCheck.cpp
  AssignOperatorSignatureCheck.cpp
  BoolPointerImplicitConversion.cpp
  InaccurateEraseCheck.cpp
  InefficientAlgorithmCheck.cpp
  MiscTidyModule.cpp
  StaticAssertCheck.cpp
  SwappedArgumentsCheck.cpp
  UndelegatedConstructor.cpp
  UnusedRAII.cpp
  UniqueptrResetRelease.cpp
  UseOverride.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
