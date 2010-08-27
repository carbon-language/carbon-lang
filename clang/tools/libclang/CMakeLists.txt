set(SHARED_LIBRARY TRUE)

set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS
  clangFrontend
  clangDriver
  clangSerialization
  clangParse
  clangSema
  clangAnalysis
  clangAST
  clangLex
  clangBasic)

set( LLVM_LINK_COMPONENTS
  bitreader
  mc
  core
  )

add_clang_library(libclang
  CIndex.cpp
  CIndexCXX.cpp
  CIndexCodeCompletion.cpp
  CIndexDiagnostic.cpp
  CIndexInclusionStack.cpp
  CIndexUSRs.cpp
  CIndexer.cpp
  CXCursor.cpp
  CXType.cpp
  ../../include/clang-c/Index.h
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # dylib versioning information
  # FIXME: Is there a more CMake-ish way to handle this?
  set(LIBCLANG_VERSION 1 
      CACHE STRING "Version number of the libclang library")
  set(LIBCLANG_SUBVERSION 0
      CACHE STRING "Minor version number of the libclang library")
  set(LIBCLANG_LINK_FLAGS 
      "-Wl,-current_version -Wl,${LIBCLANG_VERSION}.${LIBCLANG_SUBVERSION} -Wl,-compatibility_version -Wl,1")

  set(LIBCLANG_LINK_FLAGS 
      "${LIBCLANG_LINK_FLAGS} -Wl,-dead_strip -Wl,-seg1addr -Wl,0xE0000000")

  set_target_properties(libclang
    PROPERTIES
    LINK_FLAGS "${LIBCLANG_LINK_FLAGS}"
    INSTALL_NAME_DIR "@executable_path/../lib")
endif()

if(MSVC)
  # windows.h doesn't compile with /Za
  get_target_property(NON_ANSI_COMPILE_FLAGS libclang COMPILE_FLAGS)
  string(REPLACE /Za "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  set_target_properties(libclang PROPERTIES COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
endif(MSVC)

set_target_properties(libclang
  PROPERTIES
  LINKER_LANGUAGE CXX
  DEFINE_SYMBOL _CINDEX_LIB_)
