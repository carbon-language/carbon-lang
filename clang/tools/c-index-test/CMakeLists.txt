set( LLVM_LINK_COMPONENTS
  support
  mc
  )

add_clang_executable(c-index-test
  c-index-test.c
  )

target_link_libraries(c-index-test
  libclang
  )

set_target_properties(c-index-test
  PROPERTIES
  LINKER_LANGUAGE CXX)

install(TARGETS c-index-test RUNTIME DESTINATION bin)
