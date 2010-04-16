set(SHARED_LIBRARY TRUE)

set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS
 clangFrontend
  clangDriver
  clangSema
  clangAnalysis
  clangAST
  clangParse
  clangLex
  clangBasic)

set( LLVM_LINK_COMPONENTS
  bitreader
  mc
  core
  )

add_clang_library(CIndex
  CIndex.cpp
  CIndexCodeCompletion.cpp
  CIndexDiagnostic.cpp
  CIndexInclusionStack.cpp
  CIndexUSRs.cpp
  CIndexer.cpp
  CXCursor.cpp
  ../../include/clang-c/Index.h
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # FIXME: Deal with LLVM_SUBMIT_VERSION?

  # FIXME: This uses a special darwin-specific exports file in order to
  # get underscore-prefixed names. It would be better to have build rules
  # which know how to produce a darwin-suitable exports file from the
  # regular exports file.
  set_target_properties(CIndex
    PROPERTIES
    LINK_FLAGS "-avoid-version -Wl,-exported_symbols_list -Wl,${CMAKE_CURRENT_SOURCE_DIR}/CIndex.darwin.exports -Wl,-dead_strip -Wl,-seg1addr -Wl,0xE0000000"
    INSTALL_NAME_DIR "@executable_path/../lib"
    )
endif()

if(MSVC)
  # windows.h doesn't compile with /Za
  get_target_property(NON_ANSI_COMPILE_FLAGS CIndex COMPILE_FLAGS)
  string(REPLACE /Za "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  set_target_properties(CIndex PROPERTIES COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
endif(MSVC)

set_target_properties(CIndex
  PROPERTIES
  LINKER_LANGUAGE CXX
  DEFINE_SYMBOL _CINDEX_LIB_)
