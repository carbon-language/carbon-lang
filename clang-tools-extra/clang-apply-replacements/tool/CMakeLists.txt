set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  support
  mc
  )

add_clang_executable(clang-apply-replacements
  ClangApplyReplacementsMain.cpp
  )
target_link_libraries(clang-apply-replacements
  clangApplyReplacements
  )

install(TARGETS clang-apply-replacements
  RUNTIME DESTINATION bin)
