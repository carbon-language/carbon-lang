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

add_llvm_loadable_module(PrintFunctionNames PrintFunctionNames.cpp)

add_dependencies(PrintFunctionNames
  ClangAttrClasses
  ClangAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangStmtNodes
  )
