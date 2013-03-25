set(LLVM_LINK_COMPONENTS mc)

add_clang_library(clangBasic
  Builtins.cpp
  CharInfo.cpp
  Diagnostic.cpp
  DiagnosticIDs.cpp
  FileManager.cpp
  FileSystemStatCache.cpp
  IdentifierTable.cpp
  LangOptions.cpp
  Module.cpp
  ObjCRuntime.cpp
  OpenMPKinds.cpp
  OperatorPrecedence.cpp
  SourceLocation.cpp
  SourceManager.cpp
  TargetInfo.cpp
  Targets.cpp
  TokenKinds.cpp
  Version.cpp
  VersionTuple.cpp
  )

# Determine Subversion revision.
# FIXME: This only gets updated when CMake is run, so this revision number
# may be out-of-date!
if( NOT IS_SYMLINK "${CLANG_SOURCE_DIR}" )  # See PR 8437
  find_package(Subversion)
endif()
if (Subversion_FOUND AND EXISTS "${CLANG_SOURCE_DIR}/.svn")
  # Create custom target to generate the Subversion version include.
  add_custom_target(clang_revision_tag  ALL
    COMMAND ${CMAKE_COMMAND} -DFIRST_SOURCE_DIR=${LLVM_MAIN_SRC_DIR}
                             -DFIRST_REPOSITORY=LLVM_REPOSITORY
                             -DSECOND_SOURCE_DIR=${CLANG_SOURCE_DIR}
                             -DSECOND_REPOSITORY=SVN_REPOSITORY
                             -DHEADER_FILE=${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc
   -P ${LLVM_MAIN_SRC_DIR}/cmake/modules/GetSVN.cmake)

  # Mark the generated header as being generated.
message(STATUS "Expecting header to go in ${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")

endif()

add_dependencies(clangBasic
  ClangARMNeon
  ClangAttrList
  ClangDiagnosticAnalysis
  ClangDiagnosticAST
  ClangDiagnosticComment
  ClangDiagnosticCommon
  ClangDiagnosticDriver
  ClangDiagnosticFrontend
  ClangDiagnosticGroups
  ClangDiagnosticIndexName
  ClangDiagnosticLex
  ClangDiagnosticParse
  ClangDiagnosticSema
  ClangDiagnosticSerialization
  )

# clangBasic depends on the version.
if (Subversion_FOUND AND EXISTS "${CLANG_SOURCE_DIR}/.svn")
  add_dependencies(clangBasic clang_revision_tag)
endif()