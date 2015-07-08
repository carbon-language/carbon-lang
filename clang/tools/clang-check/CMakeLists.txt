set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  )

set(LLVM_NO_DEAD_STRIP 1)

add_clang_executable(clang-check
  ClangCheck.cpp
  )

target_link_libraries(clang-check
  clangAST
  clangCodeGen
  clangDriver
  clangFrontend
  clangRewriteFrontend
  clangStaticAnalyzerFrontend
  clangTooling
  )

install(TARGETS clang-check
  RUNTIME DESTINATION bin)
