set(LLVM_LINK_COMPONENTS
  Support
  )

# FIXME: Get rid of private headers in other modules.
# It is for convenience of clang/lib/StaticAnalyzer.
include_directories(${CLANG_BINARY_DIR} ${CLANG_SOURCE_DIR})

add_clang_library(clangTidy
  ClangTidy.cpp
  ClangTidyModule.cpp
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
