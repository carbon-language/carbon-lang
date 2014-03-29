include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMARM64Info
  ARM64TargetInfo.cpp
  )

add_dependencies(LLVMARM64Info ARM64CommonTableGen)
