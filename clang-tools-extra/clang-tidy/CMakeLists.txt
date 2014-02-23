set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangTidy
  ClangTidy.cpp
  ClangTidyModule.cpp
  ClangTidyDiagnosticConsumer.cpp

  DEPENDS
  ClangSACheckers
  )
target_link_libraries(clangTidy
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangRewriteCore
  clangStaticAnalyzerFrontend
  clangTooling
  )

add_subdirectory(tool)
add_subdirectory(llvm)
add_subdirectory(google)
