set(SHARED_LIBRARY TRUE)

set(LLVM_USED_LIBS
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

add_clang_library(libclang
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

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Darwin-specific linker flags

  set(LIBCLANG_LINK_FLAGS "-Wl,-compatibility_version -Wl,1")
                        
  set(LIBCLANG_LINK_FLAGS 
      "${LIBCLANG_LINK_FLAGS} -Wl,-dead_strip -Wl,-seg1addr -Wl,0xE0000000")

  set_target_properties(libclang
    PROPERTIES
    LINK_FLAGS "${LIBCLANG_LINK_FLAGS}"
    INSTALL_NAME_DIR "@executable_path/../lib")
endif()

# Versioning information
set_target_properties(libclang PROPERTIES VERSION ${LIBCLANG_LIBRARY_VERSION})

if(MSVC)
  # windows.h doesn't compile with /Za
  get_target_property(NON_ANSI_COMPILE_FLAGS libclang COMPILE_FLAGS)
  string(REPLACE /Za "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  set_target_properties(libclang PROPERTIES COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
endif(MSVC)

set_target_properties(libclang
  PROPERTIES
  PREFIX ""  # Otherwise we get liblibclang.so
  LINKER_LANGUAGE CXX
  DEFINE_SYMBOL _CINDEX_LIB_)
