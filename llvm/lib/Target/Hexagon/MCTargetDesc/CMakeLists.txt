add_llvm_library(LLVMHexagonDesc
  HexagonMCAsmInfo.cpp
  HexagonMCCodeEmitter.cpp
  HexagonMCInst.cpp
  HexagonMCTargetDesc.cpp
  )

add_dependencies(LLVMHexagonDesc HexagonCommonTableGen)