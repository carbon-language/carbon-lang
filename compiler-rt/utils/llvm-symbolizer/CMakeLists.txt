# FIXME: As we plan to execute llvm-symbolizer binary from Sanitizer
# runtime, it has to be compiled for all supported targets (x86_64, i386 etc).
# This means that we need LLVM libraries to be compiled for these
# targets as well. Currently, there is no support for such a build strategy.

set(LLVM_LINK_COMPONENTS
  DebugInfo
  Object
  )

add_llvm_tool(llvm-symbolizer
  llvm-symbolizer.cpp
  )

# FIXME: Set sane output directory for llvm-symbolizer
set_target_properties(llvm-symbolizer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
