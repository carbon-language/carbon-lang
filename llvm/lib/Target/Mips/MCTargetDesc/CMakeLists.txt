add_llvm_library(LLVMMipsDesc
  MipsMCTargetDesc.cpp
  MipsMCAsmInfo.cpp
  )
add_dependencies(LLVMMipsDesc MipsCommonTableGen)

target_link_libraries(LLVMMipsDesc LLVMMipsAsmPrinter)
