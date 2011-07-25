add_subdirectory(src)

# TODO: support plugins and user-configured builds.
# See ./doc/LLVMC-Reference.rst "Customizing LLVMC: the compilation graph"

if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()
