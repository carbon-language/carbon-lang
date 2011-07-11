set(LLVM_USED_LIBS
  clangARCMigrate
  clangRewrite
  clangFrontend
  clangDriver
  clangSerialization
  clangIndex
  clangSema
  clangAST
  clangLex
  clangBasic)

set( LLVM_LINK_COMPONENTS
  support
  mc
  )

set(SOURCES
  ARCMigrate.cpp
  CIndex.cpp
  CIndexCXX.cpp
  CIndexCodeCompletion.cpp
  CIndexDiagnostic.cpp
  CIndexInclusionStack.cpp
  CIndexUSRs.cpp
  CIndexer.cpp
  CXCursor.cpp
  CXString.cpp
  CXType.cpp
  ../../include/clang-c/Index.h
  )

if( LLVM_ENABLE_PIC )
  set(SHARED_LIBRARY TRUE)
  add_clang_library(libclang ${SOURCES})

  set_target_properties(libclang
    PROPERTIES
    OUTPUT_NAME "libclang"
    VERSION ${LIBCLANG_LIBRARY_VERSION}
    DEFINE_SYMBOL _CINDEX_LIB_)

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(LIBCLANG_LINK_FLAGS
      "-Wl,-compatibility_version -Wl,1 -Wl,-dead_strip -Wl,-seg1addr -Wl,0xE0000000")
    set_target_properties(libclang
      PROPERTIES
      LINK_FLAGS "${LIBCLANG_LINK_FLAGS}"
      INSTALL_NAME_DIR "@executable_path/../lib")
  endif()

  if(MSVC)
    # windows.h doesn't compile with /Za
    get_target_property(NON_ANSI_COMPILE_FLAGS libclang COMPILE_FLAGS)
    string(REPLACE "/Za" "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
    set_target_properties(libclang PROPERTIES
      COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  endif()

  set(LIBCLANG_STATIC_TARGET_NAME libclang_static)
else()
  set(LIBCLANG_STATIC_TARGET_NAME libclang)
endif()

if( NOT BUILD_SHARED_LIBS AND NOT WIN32 )
  add_clang_library(${LIBCLANG_STATIC_TARGET_NAME} STATIC ${SOURCES})

  set_target_properties(${LIBCLANG_STATIC_TARGET_NAME}
    PROPERTIES
    OUTPUT_NAME "libclang")
endif()
