include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMBlackfinInfo
  BlackfinTargetInfo.cpp
  )

add_llvm_library_dependencies(LLVMBlackfinInfo
  LLVMMC
  LLVMSupport
  LLVMTarget
  )

add_dependencies(LLVMBlackfinInfo BlackfinCommonTableGen)
