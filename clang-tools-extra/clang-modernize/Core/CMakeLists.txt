set(LLVM_LINK_COMPONENTS support)

add_clang_library(modernizeCore
  ReplacementHandling.cpp
  Transforms.cpp
  Transform.cpp
  IncludeExcludeInfo.cpp
  PerfSupport.cpp
  IncludeDirectives.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangTooling
  )
