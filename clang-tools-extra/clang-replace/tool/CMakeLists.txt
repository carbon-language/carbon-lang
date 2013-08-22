set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  support
  mc
  )

add_clang_executable(clang-replace
  ClangReplaceMain.cpp
  )
target_link_libraries(clang-replace
  clangReplace
  )

install(TARGETS clang-replace
  RUNTIME DESTINATION bin)
