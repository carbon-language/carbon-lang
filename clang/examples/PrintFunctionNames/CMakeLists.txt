set( LLVM_LINK_COMPONENTS
  Support
  )

# If we don't need RTTI or EH, there's no reason to export anything
# from the plugin.
if( NOT WIN32 ) # Win32 mangles symbols differently, and
                # PrintFunctionNames.export contains C++ symbols.
  if( NOT LLVM_REQUIRES_RTTI )
    if( NOT LLVM_REQUIRES_EH )
      set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/PrintFunctionNames.exports)
    endif()
  endif()
endif()

add_clang_library(PrintFunctionNames MODULE PrintFunctionNames.cpp)

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
