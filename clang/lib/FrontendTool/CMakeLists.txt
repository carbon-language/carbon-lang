set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  ClangDriverOptions
  )

target_link_libraries(clangFrontendTool
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangRewriteFrontend
  )

if(CLANG_ENABLE_ARCMT)
  target_link_libraries(clangFrontendTool
    clangARCMigrate
    )
endif()

if(CLANG_ENABLE_STATIC_ANALYZER)
  target_link_libraries(clangFrontendTool
    clangStaticAnalyzerFrontend
    )
endif()
