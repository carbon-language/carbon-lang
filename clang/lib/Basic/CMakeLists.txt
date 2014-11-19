set(LLVM_LINK_COMPONENTS
  Core
  MC
  Support
  )

# Figure out if we can track VC revisions.
function(find_first_existing_file out_var)
  foreach(file ${ARGN})
    if(EXISTS "${file}")
      set(${out_var} "${file}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

find_first_existing_file(llvm_vc
  "${LLVM_MAIN_SRC_DIR}/.git/logs/HEAD"
  "${LLVM_MAIN_SRC_DIR}/.svn/entries")
find_first_existing_file(clang_vc
  "${CLANG_SOURCE_DIR}/.git/logs/HEAD"
  "${CLANG_SOURCE_DIR}/.svn/entries")

if(DEFINED llvm_vc AND DEFINED clang_vc)
  # Create custom target to generate the VC revision include.
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc"
    DEPENDS "${llvm_vc}" "${clang_vc}"
    COMMAND
    ${CMAKE_COMMAND} "-DFIRST_SOURCE_DIR=${LLVM_MAIN_SRC_DIR}"
                     "-DFIRST_NAME=LLVM"
                     "-DSECOND_SOURCE_DIR=${CLANG_SOURCE_DIR}"
                     "-DSECOND_NAME=SVN"
                     "-DHEADER_FILE=${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc"
                     -P "${LLVM_MAIN_SRC_DIR}/cmake/modules/GetSVN.cmake")

  # Mark the generated header as being generated.
  set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc"
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")
  set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc")
else()
  set(version_inc)
endif()

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
  SanitizerBlacklist.cpp
  Sanitizers.cpp
  SourceLocation.cpp
  SourceManager.cpp
  TargetInfo.cpp
  Targets.cpp
  TokenKinds.cpp
  Version.cpp
  VersionTuple.cpp
  VirtualFileSystem.cpp
  Warnings.cpp
  ${version_inc}
  )

