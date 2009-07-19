include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMIA64Info
  IA64TargetInfo.cpp
  )

add_dependencies(LLVMIA64Info IA64CodeGenTable_gen)
