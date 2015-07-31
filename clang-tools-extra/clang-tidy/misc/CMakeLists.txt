set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyMiscModule
  ArgumentCommentCheck.cpp
  AssertSideEffectCheck.cpp
  AssignOperatorSignatureCheck.cpp
  BoolPointerImplicitConversionCheck.cpp
  InaccurateEraseCheck.cpp
  InefficientAlgorithmCheck.cpp
  MacroParenthesesCheck.cpp
  MacroRepeatedSideEffectsCheck.cpp
  MiscTidyModule.cpp
  NoexceptMoveConstructorCheck.cpp
  StaticAssertCheck.cpp
  SwappedArgumentsCheck.cpp
  UndelegatedConstructor.cpp
  UnusedAliasDeclsCheck.cpp
  UnusedParametersCheck.cpp
  UnusedRAIICheck.cpp
  UniqueptrResetReleaseCheck.cpp
  UseOverrideCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  )
