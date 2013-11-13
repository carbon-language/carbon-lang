set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  bitreader
  mc
  )

set(SOURCES
  ARCMigrate.cpp
  CIndex.cpp
  CIndexCXX.cpp
  CIndexCodeCompletion.cpp
  CIndexDiagnostic.cpp
  CIndexDiagnostic.h
  CIndexHigh.cpp
  CIndexInclusionStack.cpp
  CIndexUSRs.cpp
  CIndexer.cpp
  CIndexer.h
  CXComment.cpp
  CXCursor.cpp
  CXCursor.h
  CXCompilationDatabase.cpp
  CXLoadedDiagnostic.cpp
  CXLoadedDiagnostic.h
  CXSourceLocation.cpp
  CXSourceLocation.h
  CXStoredDiagnostic.cpp
  CXString.cpp
  CXString.h
  CXTranslationUnit.h
  CXType.cpp
  CXType.h
  IndexBody.cpp
  IndexDecl.cpp
  IndexTypeSourceInfo.cpp
  Index_Internal.h
  Indexing.cpp
  IndexingContext.cpp
  IndexingContext.h
  ../../include/clang-c/Index.h
  )

set(LIBRARIES
  clangIndex
  clangARCMigrate
  clangRewriteCore
  clangRewriteFrontend
  clangFrontend
  clangDriver
  clangSerialization
  clangSema
  clangEdit
  clangAST
  clangLex
  clangTooling
  clangBasic
  )

set(GENERATED_HEADERS
  ClangAttrClasses
  ClangAttrList
  ClangAttrParsedAttrList
  ClangCommentNodes
  ClangDiagnosticCommon
  ClangDiagnosticFrontend
  ClangDeclNodes
  ClangStmtNodes
  )

set(EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/libclang.exports)

if( LLVM_ENABLE_PIC )
  set(SHARED_LIBRARY TRUE)
  add_clang_library(libclang ${SOURCES})
  target_link_libraries(libclang ${LIBRARIES})
  add_dependencies(libclang ${GENERATED_HEADERS} clang-headers)

  if(WIN32)
    set_target_properties(libclang
      PROPERTIES
      OUTPUT_NAME "libclang"
      VERSION ${LIBCLANG_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
  else()
    set_target_properties(libclang
      PROPERTIES
      OUTPUT_NAME "clang"
      VERSION ${LIBCLANG_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
  endif()

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(LIBCLANG_LINK_FLAGS
      " -Wl,-compatibility_version -Wl,1 -Wl,-dead_strip")
    if (DEFINED ${LLVM_SUBMIT_VERSION})
      set(LIBCLANG_LINK_FLAGS
        "${LIBCLANG_LINK_FLAGS} -Wl,-current_version -Wl,${LLVM_SUBMIT_VERSION}.${LLVM_SUBMIT_SUBVERSION}")
    endif()

    set_property(TARGET libclang APPEND_STRING PROPERTY
                 LINK_FLAGS ${LIBCLANG_LINK_FLAGS})
    set_target_properties(libclang
      PROPERTIES
      INSTALL_NAME_DIR "@rpath")
  endif()


  set(LIBCLANG_STATIC_TARGET_NAME libclang_static)
else()
  set(LIBCLANG_STATIC_TARGET_NAME libclang)
endif()

option(LIBCLANG_BUILD_STATIC
  "Build libclang as a static library (in addition to a shared one)" OFF)

if( (NOT LLVM_ENABLE_PIC OR LIBCLANG_BUILD_STATIC) AND NOT WIN32 )
  add_clang_library(${LIBCLANG_STATIC_TARGET_NAME} STATIC ${SOURCES})
  target_link_libraries(${LIBCLANG_STATIC_TARGET_NAME} ${LIBRARIES})
  add_dependencies(${LIBCLANG_STATIC_TARGET_NAME} ${GENERATED_HEADERS} clang-headers)

  set_target_properties(${LIBCLANG_STATIC_TARGET_NAME}
    PROPERTIES
    OUTPUT_NAME "clang")
endif()
