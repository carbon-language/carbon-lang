set(SHARED_LIBRARY TRUE)

set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS
  clangIndex
  clangFrontend
  clangDriver
  clangSema
  clangAnalysis
  clangAST
  clangParse
  clangLex
  clangBasic)

set( LLVM_LINK_COMPONENTS
  bitreader
  mc
  core
  )

add_clang_library(PrintFunctionNames PrintFunctionNames.cpp)

set_target_properties(PrintFunctionNames
  PROPERTIES
  LINKER_LANGUAGE CXX)
