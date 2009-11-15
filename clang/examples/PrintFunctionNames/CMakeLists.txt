set(SHARED_LIBRARY TRUE)

set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS
  clangFrontend clangIndex clangSema clangAnalysis clangAST clangParse clangLex clangBasic)

set( LLVM_LINK_COMPONENTS
  MC
  support
  )

add_clang_library(PrintFunctionNames PrintFunctionNames.cpp)

set_target_properties(PrintFunctionNames
  PROPERTIES
  LINKER_LANGUAGE CXX)
