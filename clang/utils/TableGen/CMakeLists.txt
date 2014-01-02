set(LLVM_LINK_COMPONENTS Support)

add_tablegen(clang-tblgen CLANG
  ClangASTNodesEmitter.cpp
  ClangAttrEmitter.cpp
  ClangCommentCommandInfoEmitter.cpp
  ClangCommentHTMLNamedCharacterReferenceEmitter.cpp
  ClangCommentHTMLTagsEmitter.cpp
  ClangDiagnosticsEmitter.cpp
  ClangSACheckersEmitter.cpp
  NeonEmitter.cpp
  TableGen.cpp
  )
set_output_directory(clang-tblgen ${CLANG_RUNTIME_OUTPUT_INTDIR} ${CLANG_LIBRARY_OUTPUT_INTDIR})
