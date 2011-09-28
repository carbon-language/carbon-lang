set(LLVM_USED_LIBS libclang)

set( LLVM_LINK_COMPONENTS
  support
  mc
  )

add_clang_executable(c-index-test
  c-index-test.c
  )

set_target_properties(c-index-test
  PROPERTIES
  LINKER_LANGUAGE CXX)

install(TARGETS c-index-test RUNTIME DESTINATION bin)
