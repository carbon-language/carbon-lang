set(LLVM_LINK_COMPONENTS
  BitReader
  Support
  )

set(SOURCES
  ARCMigrate.cpp
  BuildSystem.cpp
  CIndex.cpp
  CIndexCXX.cpp
  CIndexCodeCompletion.cpp
  CIndexDiagnostic.cpp
  CIndexHigh.cpp
  CIndexInclusionStack.cpp
  CIndexUSRs.cpp
  CIndexer.cpp
  CXComment.cpp
  CXCursor.cpp
  CXCompilationDatabase.cpp
  CXLoadedDiagnostic.cpp
  CXSourceLocation.cpp
  CXStoredDiagnostic.cpp
  CXString.cpp
  CXType.cpp
  IndexBody.cpp
  IndexDecl.cpp
  IndexTypeSourceInfo.cpp
  Indexing.cpp
  IndexingContext.cpp

  ADDITIONAL_HEADERS
  CIndexDiagnostic.h
  CIndexer.h
  CXCursor.h
  CXLoadedDiagnostic.h
  CXSourceLocation.h
  CXString.h
  CXTranslationUnit.h
  CXType.h
  Index_Internal.h
  IndexingContext.h
  ../../include/clang-c/Index.h
  )

set(LIBRARIES
  clangARCMigrate
  clangAST
  clangBasic
  clangFrontend
  clangIndex
  clangLex
  clangSema
  clangTooling
  )

option(LIBCLANG_BUILD_STATIC
  "Build libclang as a static library (in addition to a shared one)" OFF)

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/libclang.exports)

if(MSVC)
  # Avoid LNK4197 not to spceify libclang.def here.
  # Each functions is exported as "dllexport" in include/clang-c.
  # KB835326
  set(LLVM_EXPORTED_SYMBOL_FILE)
endif()

if( LLVM_ENABLE_PIC )
  add_clang_library(libclang SHARED
    ${SOURCES}
    DEPENDS clang-headers
    )
  target_link_libraries(libclang ${LIBRARIES})

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
    set(LIBCLANG_LINK_FLAGS " -Wl,-compatibility_version -Wl,1")
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

if( (NOT LLVM_ENABLE_PIC OR LIBCLANG_BUILD_STATIC) AND NOT WIN32 )
  add_clang_library(${LIBCLANG_STATIC_TARGET_NAME} STATIC ${SOURCES})
  target_link_libraries(${LIBCLANG_STATIC_TARGET_NAME} ${LIBRARIES})

  set_target_properties(${LIBCLANG_STATIC_TARGET_NAME}
    PROPERTIES
    OUTPUT_NAME "clang")
endif()
