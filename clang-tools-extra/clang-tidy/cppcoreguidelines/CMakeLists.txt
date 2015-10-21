set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyCppCoreGuidelinesModule
  CppCoreGuidelinesTidyModule.cpp
  ProBoundsPointerArithmeticCheck.cpp
  ProTypeConstCastCheck.cpp
  ProTypeReinterpretCastCheck.cpp
  ProTypeStaticCastDowncastCheck.cpp
  ProTypeUnionAccessCheck.cpp
  ProTypeVarargCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyMiscModule
  clangTidyUtils
  clangTooling
  )
