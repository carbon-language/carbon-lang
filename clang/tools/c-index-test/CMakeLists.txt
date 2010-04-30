set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  libclang
  clangIndex
  clangFrontend
  clangDriver
  clangSema
  clangAnalysis
  clangAST
  clangParse
  clangLex
  clangBasic
  )

set( LLVM_LINK_COMPONENTS
  bitreader
  mc
  core
  )

add_clang_executable(c-index-test
  c-index-test.c
  )

set_target_properties(c-index-test
  PROPERTIES
  LINKER_LANGUAGE CXX)

