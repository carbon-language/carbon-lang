set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangTidy
  ClangTidy.cpp
  ClangTidyModule.cpp
  ClangTidyDiagnosticConsumer.cpp
  ClangTidyOptions.cpp

  DEPENDS
  ClangSACheckers

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangRewrite
  clangSema
  clangStaticAnalyzerCore
  clangStaticAnalyzerFrontend
  clangTooling
  clangToolingCore
  )

add_subdirectory(tool)
add_subdirectory(llvm)
add_subdirectory(google)
add_subdirectory(misc)
add_subdirectory(readability)
add_subdirectory(utils)
