include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMXCoreInfo
  XCoreTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMXCoreInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMXCoreInfo XCoreCodeGenTable_gen)
