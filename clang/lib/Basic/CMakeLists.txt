set(LLVM_NO_RTTI 1)

add_clang_library(clangBasic
  Builtins.cpp
  ConvertUTF.c
  Diagnostic.cpp
  FileManager.cpp
  IdentifierTable.cpp
  SourceLocation.cpp
  SourceManager.cpp
  TargetInfo.cpp
  Targets.cpp
  TokenKinds.cpp
  Version.cpp
  )

# Determine Subversion revision.
# FIXME: This only gets updated when CMake is run, so this revision number
# may be out-of-date!
if( NOT IS_SYMLINK "${CLANG_SOURCE_DIR}" )  # See PR 8437
  find_package(Subversion)
endif()
if (Subversion_FOUND AND EXISTS "${CLANG_SOURCE_DIR}/.svn")
  Subversion_WC_INFO(${CLANG_SOURCE_DIR} CLANG)
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "SVN_REVISION=\"${CLANG_WC_REVISION}\"")
endif()

add_dependencies(clangBasic 
                 ClangARMNeon
                 ClangAttrList
                 ClangDiagnosticAnalysis
                 ClangDiagnosticAST
                 ClangDiagnosticCommon
                 ClangDiagnosticDriver
                 ClangDiagnosticFrontend
                 ClangDiagnosticGroups
                 ClangDiagnosticLex
                 ClangDiagnosticParse
                 ClangDiagnosticSema)
                 
