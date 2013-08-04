set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  support
  mc
  )

add_clang_executable(clang-tidy
  ClangTidyMain.cpp
  )
target_link_libraries(clang-tidy
  clangTidy
  clangTidyLLVMModule
  clangTidyGoogleModule
  )

install(TARGETS clang-tidy
  RUNTIME DESTINATION bin)

