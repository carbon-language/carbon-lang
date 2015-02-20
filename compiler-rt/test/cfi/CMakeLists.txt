configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

set(CFI_TEST_DEPS)
if(NOT COMPILER_RT_STANDALONE_BUILD)
  list(APPEND CFI_TEST_DEPS
    FileCheck
    clang
    not
  )
  if(LLVM_ENABLE_PIC AND LLVM_BINUTILS_INCDIR)
    list(APPEND CFI_TEST_DEPS
      LLVMgold
    )
  endif()
endif()

add_lit_testsuite(check-cfi "Running the cfi regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CFI_TEST_DEPS})
set_target_properties(check-cfi PROPERTIES FOLDER "Tests")
