set(MODULE TRUE)

set( LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(PrintFunctionNames PrintFunctionNames.cpp)

add_dependencies(PrintFunctionNames
  ClangAttrClasses
  ClangAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangStmtNodes
  )

target_link_libraries(PrintFunctionNames
  clangAST
  clangBasic
  clangFrontend
  )

set_target_properties(PrintFunctionNames
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")
