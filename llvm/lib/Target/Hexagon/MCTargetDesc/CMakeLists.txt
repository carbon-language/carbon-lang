add_llvm_library(LLVMHexagonDesc
  HexagonAsmBackend.cpp
  HexagonELFObjectWriter.cpp
  HexagonInstPrinter.cpp
  HexagonMCAsmInfo.cpp
  HexagonMCCodeEmitter.cpp
  HexagonMCInstrInfo.cpp
  HexagonMCShuffler.cpp
  HexagonMCTargetDesc.cpp
  HexagonShuffler.cpp
  )

add_dependencies(LLVMHexagonDesc HexagonCommonTableGen)
