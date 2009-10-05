set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  CIndex
  clangIndex
  clangFrontend
  clangSema
  clangAST
  clangLex
  clangBasic
  )

set( LLVM_LINK_COMPONENTS
  bitreader
  mc
  )

add_clang_executable(c-index-test
  c-index-test.c
  )

set_target_properties(c-index-test
  PROPERTIES
  LINKER_LANGUAGE CXX)

