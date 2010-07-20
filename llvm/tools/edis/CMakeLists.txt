set(LLVM_NO_RTTI 1)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_llvm_library(EnhancedDisassembly
  ../../include/llvm-c/EnhancedDisassembly.h
)

set_target_properties(EnhancedDisassembly
  PROPERTIES
  LINKER_LANGUAGE CXX)

