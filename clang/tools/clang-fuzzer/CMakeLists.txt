if( LLVM_USE_SANITIZE_COVERAGE )
  set(LLVM_LINK_COMPONENTS support)

  add_clang_executable(clang-fuzzer
    EXCLUDE_FROM_ALL
    ClangFuzzer.cpp
    )

  target_link_libraries(clang-fuzzer
    ${CLANG_FORMAT_LIB_DEPS}
    clangAST
    clangBasic
    clangDriver
    clangFrontend
    clangRewriteFrontend
    clangStaticAnalyzerFrontend
    clangTooling
    LLVMFuzzer
    )
endif()