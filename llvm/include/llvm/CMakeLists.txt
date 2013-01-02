add_subdirectory(IR)

if( MSVC_IDE OR XCODE )
  # Creates a dummy target containing all headers for the benefit of
  # Visual Studio users.
  file(GLOB_RECURSE headers *.h)
  add_td_sources(headers)
  add_library(llvm_headers_do_not_build EXCLUDE_FROM_ALL
    # We need at least one source file:
    ${LLVM_MAIN_SRC_DIR}/lib/Transforms/Hello/Hello.cpp
    ${headers})
  set_target_properties(llvm_headers_do_not_build PROPERTIES FOLDER "Misc")
endif()
