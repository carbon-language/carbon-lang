set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyLLVMModule
  LLVMTidyModule.cpp
  )
target_link_libraries(clangTidyLLVMModule
  clangTidy
  clangTooling
  clangBasic
  clangASTMatchers
  )
