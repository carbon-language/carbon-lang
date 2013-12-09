set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangTidy
  ClangTidy.cpp
  ClangTidyModule.cpp
  )
target_link_libraries(clangTidy
  clangTooling
  clangBasic
  clangRewriteFrontend
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  )

add_subdirectory(tool)
add_subdirectory(llvm)
add_subdirectory(google)
