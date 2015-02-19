add_llvm_library(LLVMHexagonDesc
  HexagonAsmBackend.cpp
  HexagonELFObjectWriter.cpp
  HexagonInstPrinter.cpp
  HexagonMCAsmInfo.cpp
  HexagonMCCodeEmitter.cpp
  HexagonMCInstrInfo.cpp
  HexagonMCTargetDesc.cpp
  )

add_dependencies(LLVMHexagonDesc HexagonCommonTableGen)
