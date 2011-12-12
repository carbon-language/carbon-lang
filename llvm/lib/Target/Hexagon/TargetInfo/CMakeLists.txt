include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. 
                     ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMHexagonInfo
  HexagonTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMHexagonInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMHexagonInfo HexagonCommonTableGen)
