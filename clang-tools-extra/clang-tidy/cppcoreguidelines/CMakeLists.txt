set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyCppCoreGuidelinesModule
  CppCoreGuidelinesTidyModule.cpp
  ProBoundsPointerArithmeticCheck.cpp
  ProTypeConstCastCheck.cpp
  ProTypeReinterpretCastCheck.cpp
  ProTypeStaticCastDowncastCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyCERTModule
  clangTidyMiscModule
  clangTidyUtils
  clangTooling
  )
