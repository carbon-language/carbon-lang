set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
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

add_clang_executable(index-test
  index-test.cpp
  )
