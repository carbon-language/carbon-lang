set(MODULE TRUE)

set(LLVM_NO_RTTI 1)

set( LLVM_USED_LIBS
  clangFrontendTool
  clangFrontend
  clangDriver
  clangSerialization
  clangCodeGen
  clangParse
  clangSema
  clangGRCore
  clangAnalysis
  clangIndex
  clangRewrite
  clangAST
  clangLex
  clangBasic
  )

# Why do we have to link to all this just to print out function names?
set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  codegen
  ipo
  selectiondag
  )

add_clang_library(PrintFunctionNames PrintFunctionNames.cpp)

set_target_properties(PrintFunctionNames
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
