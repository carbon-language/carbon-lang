set(LLVM_USED_LIBS clangTooling clangBasic clangAST)

add_clang_executable(remove-cstr-calls
  RemoveCStrCalls.cpp
  )
