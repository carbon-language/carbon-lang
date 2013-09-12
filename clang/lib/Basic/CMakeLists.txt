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
  ${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc
  )

# Create custom target to generate the Subversion version include.
add_custom_target(clang_revision_tag
  ${CMAKE_COMMAND} -DFIRST_SOURCE_DIR=${LLVM_MAIN_SRC_DIR}
                   -DFIRST_NAME=LLVM
                   -DSECOND_SOURCE_DIR=${CLANG_SOURCE_DIR}
                   -DSECOND_NAME=SVN
                   -DHEADER_FILE=${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc
                   -P ${LLVM_MAIN_SRC_DIR}/cmake/modules/GetSVN.cmake)

# Mark the generated header as being generated.
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

# Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
set_source_files_properties(Version.cpp
  PROPERTIES COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")

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
  clang_revision_tag
)

# Force regeneration now.
FILE(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc)

if(CLANG_ALWAYS_CHECK_VC_REV)
  add_custom_command(TARGET clangBasic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove
            ${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc)
endif()
