add_llvm_library(LLVMMipsDesc
  MipsAsmBackend.cpp
  MipsMCAsmInfo.cpp
  MipsMCCodeEmitter.cpp
  MipsMCTargetDesc.cpp
  MipsELFObjectWriter.cpp
  MipsReginfo.cpp
  MipsELFStreamer.cpp
  )

add_dependencies(LLVMMipsDesc MipsCommonTableGen)
