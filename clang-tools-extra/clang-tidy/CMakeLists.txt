set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  support
  mc
  )

add_clang_library(clangTidy
  ClangTidy.cpp
  ClangTidyModule.cpp
  )
target_link_libraries(clangTidy
  clangTooling
  clangBasic
  clangRewriteFrontend
  clangTidyLLVMModule
  clangTidyGoogleModule
  )

add_subdirectory(tool)
add_subdirectory(llvm)
add_subdirectory(google)
