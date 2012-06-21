set(MODULE TRUE)

set( LLVM_LINK_COMPONENTS support mc)

add_clang_library(SampleAnalyzerPlugin MainCallChecker.cpp)

target_link_libraries(SampleAnalyzerPlugin
  clangStaticAnalyzerCore
  )

set_target_properties(SampleAnalyzerPlugin
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
