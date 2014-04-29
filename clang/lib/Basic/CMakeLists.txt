set(LLVM_LINK_COMPONENTS
  MC
  Support
  )

add_clang_library(clangBasic
  Attributes.cpp
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
  VirtualFileSystem.cpp
  Warnings.cpp
  )

# Determine Subversion revision.
# FIXME: This only gets updated when CMake is run, so this revision number
# may be out-of-date!
if( NOT IS_SYMLINK "${CLANG_SOURCE_DIR}" )  # See PR 8437
  find_package(Subversion)
endif()
if (Subversion_FOUND AND EXISTS "${CLANG_SOURCE_DIR}/.svn")
  set(FIRST_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(FIRST_REPOSITORY LLVM_REPOSITORY)
  set(SECOND_SOURCE_DIR ${CLANG_SOURCE_DIR})
  set(SECOND_REPOSITORY SVN_REPOSITORY)
  set(HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc)
  include(GetSVN)

  # Mark the generated header as being generated.
  message(STATUS "Expecting header to go in ${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc")
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")

endif()
