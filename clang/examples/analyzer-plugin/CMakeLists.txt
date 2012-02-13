set(MODULE TRUE)

set( LLVM_USED_LIBS
  clangStaticAnalyzerCore
  )

set( LLVM_LINK_COMPONENTS support mc)

add_clang_library(SampleAnalyzerPlugin MainCallChecker.cpp)

set_target_properties(SampleAnalyzerPlugin
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
