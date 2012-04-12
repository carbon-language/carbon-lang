add_llvm_library(LLVMHexagonDesc
  HexagonMCAsmInfo.cpp
  HexagonMCTargetDesc.cpp
  )

add_dependencies(LLVMHexagonDesc HexagonCommonTableGen)
