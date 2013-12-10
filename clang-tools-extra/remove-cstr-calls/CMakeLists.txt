set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(remove-cstr-calls
  RemoveCStrCalls.cpp
  )

target_link_libraries(remove-cstr-calls
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangLex
  clangTooling
  )
