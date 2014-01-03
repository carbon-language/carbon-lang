set(LLVM_LINK_COMPONENTS
  Support
  )

# FIXME: Get rid of private headers in other modules.
# It is hack to find "../../../lib/StaticAnalyzer".
include_directories(${CLANG_SOURCE_DIR}/lib/StaticAnalyzer/Core)

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
