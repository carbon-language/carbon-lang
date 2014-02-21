include_directories( ${CMAKE_CURRENT_BINARY_DIR}/../Checkers )

set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangStaticAnalyzerFrontend
  AnalysisConsumer.cpp
  CheckerRegistration.cpp
  FrontendActions.cpp
  )

target_link_libraries(clangStaticAnalyzerFrontend
  clangAST
  clangAnalysis
  clangBasic
  clangFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  )
