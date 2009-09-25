set(SHARED_LIBRARY TRUE)

set(LLVM_NO_RTTI 1)

set(LLVM_USED_LIBS
  clangFrontend clangIndex clangSema clangAST clangLex clangBasic)

set( LLVM_LINK_COMPONENTS
  MC
  support
  )

add_clang_library(CIndex CIndex.cpp)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # FIXME: Deal with LLVM_SUBMIT_VERSION?

  set_target_properties(CIndex
    PROPERTIES
    LINK_FLAGS "-avoid-version -Wl,-exported_symbols_list -Wl,${CMAKE_CURRENT_SOURCE_DIR}/CIndex.exports -Wl,-dead_strip -Wl,-seg1addr -Wl,0xE0000000"
    INSTALL_NAME_DIR "@executable_path/../lib"
    )
endif()
