add_llvm_library(LLVMHexagonInfo
  HexagonTargetInfo.cpp
  )

add_dependencies(LLVMHexagonInfo HexagonCommonTableGen)
