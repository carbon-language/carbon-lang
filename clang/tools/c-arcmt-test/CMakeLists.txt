set(LLVM_USED_LIBS libclang)

set( LLVM_LINK_COMPONENTS
  support
  mc
  )

add_clang_executable(c-arcmt-test
  c-arcmt-test.cpp
  )

set_target_properties(c-arcmt-test
  PROPERTIES
  LINKER_LANGUAGE CXX)
