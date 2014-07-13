add_llvm_loadable_module(SampleAnalyzerPlugin MainCallChecker.cpp)

if(LLVM_ENABLE_PLUGINS AND (WIN32 OR CYGWIN))
  target_link_libraries(SampleAnalyzerPlugin ${cmake_2_8_12_PRIVATE}
    clangAnalysis
    clangAST
    clangStaticAnalyzerCore
    LLVMSupport
    )
endif()
